#!/usr/bin/env python3
"""Pipeline runner for AOS intrinsic wavefront analysis.

Reads runs.yaml, executes pending steps, and updates status.
Run from the aos/ directory.

Usage:
    python code/run_pipeline.py status                # show all runs
    python code/run_pipeline.py run                   # run all pending steps
    python code/run_pipeline.py run chunk1             # run one specific run
    python code/run_pipeline.py run chunk1 fit         # run one specific step
    python code/run_pipeline.py set chunk1 mktable done  # manually set status
    python code/run_pipeline.py reset chunk1           # reset all steps to pending
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

AOS_DIR = Path(__file__).resolve().parent.parent
RUNS_FILE = AOS_DIR / 'runs.yaml'
PARAM_SETS_FILE = AOS_DIR / 'param_sets.yaml'
# Logs live outside of output/ (which may be a symlink to a data disk)
LOG_DIR = AOS_DIR / 'logs'
LOG_FILE = LOG_DIR / 'pipeline.log'

STEP_ORDER = ['mktable', 'fit', 'plots', 'movie',
              'fit_ccs', 'plots_ccs', 'movie_ccs']
STEP_DEPS = {
    'mktable': [],
    'fit': ['mktable'],
    'plots': ['fit'],
    'movie': ['fit'],
    'fit_ccs': ['mktable'],
    'plots_ccs': ['fit_ccs'],
    'movie_ccs': ['fit_ccs'],
}
VALID_STATUSES = ['pending', 'running', 'done', 'failed', 'skip']


def _descendants_of(step):
    """Return the set of steps that transitively depend on `step`
    according to STEP_DEPS. E.g. _descendants_of('mktable') is every
    other step; _descendants_of('fit') is {'plots', 'movie'}.
    """
    from collections import defaultdict, deque
    children = defaultdict(list)
    for s, parents in STEP_DEPS.items():
        for p in parents:
            children[p].append(s)
    out = set()
    queue = deque([step])
    while queue:
        cur = queue.popleft()
        for c in children.get(cur, ()):
            if c not in out:
                out.add(c)
                queue.append(c)
    return out


def load_runs():
    with open(RUNS_FILE) as f:
        return yaml.safe_load(f)


def save_runs(data):
    with open(RUNS_FILE, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---- Concurrency-safe per-run status updates ----------------------------
#
# When several pipelines run different chunks in parallel they each have
# their own in-memory copy of runs.yaml. The naive `data = load_runs();
# data[...] = ...; save_runs(data)` pattern means whoever calls save_runs
# last clobbers the others' updates to different chunks.
#
# `update_run_status` works around that by holding an exclusive lock on a
# lock file while it re-reads the on-disk runs.yaml, applies the user's
# step-status delta to just that one run, and writes back. The lock is
# held only across the read/merge/write — milliseconds — so it's cheap
# even when many pipelines try at once.

import contextlib
import errno
import os


_LOCK_FILE = RUNS_FILE.with_suffix(RUNS_FILE.suffix + '.lock')


@contextlib.contextmanager
def _file_lock(timeout_sec=30.0):
    """Exclusive lock on runs.yaml via a sidecar .lock file and fcntl."""
    import fcntl
    import time
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(_LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"could not acquire {_LOCK_FILE} within "
                        f"{timeout_sec}s")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def update_run_status(run_name, step_status_dict, cascade=False):
    """Atomically merge a per-step status update for one run.

    Re-reads runs.yaml from disk under a file lock, applies the updates
    to `run_name` only, then writes back. This is safe when other
    pipelines are simultaneously updating *different* chunks.

    Parameters
    ----------
    run_name : str
    step_status_dict : dict[step_name, status_str]
    cascade : bool
        If True, any step set to 'pending' also forces all transitive
        descendants to 'pending'. Default False — callers like the
        pipeline runner generally don't want side effects.

    Returns the post-update step dict for the run.
    """
    with _file_lock():
        data = load_runs()
        if run_name not in data.get('runs', {}):
            raise KeyError(f"run '{run_name}' not in runs.yaml")
        run_steps = data['runs'][run_name].setdefault('steps', {})
        for step, status in step_status_dict.items():
            if status not in VALID_STATUSES:
                raise ValueError(
                    f"invalid status '{status}' for step '{step}'")
            run_steps[step] = status
            if cascade and status == 'pending':
                for desc in _descendants_of(step):
                    if desc in run_steps and run_steps[desc] != 'pending':
                        run_steps[desc] = 'pending'
        save_runs(data)
        return dict(run_steps)


def load_param_sets():
    with open(PARAM_SETS_FILE) as f:
        return yaml.safe_load(f)


def resolve_run(run_cfg, param_sets):
    """Merge param_set defaults with run-level overrides.

    Returns a dict with all resolved parameters (butler_repo, fam_collections,
    fam_programs, day_obs_min, day_obs_max, coord_sys, etc).
    """
    ps_name = run_cfg.get('param_set')
    if ps_name and ps_name in param_sets:
        resolved = dict(param_sets[ps_name])
        resolved.pop('description', None)
    else:
        resolved = {}
    # Preserve param_set name for command building
    if ps_name:
        resolved['param_set'] = ps_name
    # Run-level values override param_set defaults
    for key in ['day_obs_min', 'day_obs_max', 'coord_sys', 'butler_repo',
                'fam_collections', 'fam_programs', 'calc_intrinsics',
                'calc_mean_zernike', 'calc_focal_plane',
                'no_single_image', 'no_fit_params', 'no_trio',
                'overwrite']:
        if key in run_cfg:
            resolved[key] = run_cfg[key]
    resolved.setdefault('coord_sys', 'OCS')
    resolved.setdefault('calc_intrinsics', True)
    return resolved


def collection_phrase(resolved):
    """Derive collection phrase from resolved params (matches intrinsics_lib logic)."""
    coll = resolved['fam_collections'][0]
    parts = coll.split('/')
    if parts[0] == 'u' and len(parts) > 2:
        return parts[2]
    return coll


def donut_path(resolved):
    """Derive the donut-table parquet filename for a run."""
    phrase = collection_phrase(resolved)
    return (f"output/{phrase}_"
            f"{resolved['day_obs_min']}_{resolved['day_obs_max']}.parquet")


# Backward-compatible alias — still called hdf5_path in older notebook cells
hdf5_path = donut_path


def fits_path(resolved):
    """Derive the fits parquet filename for a run."""
    p = Path(donut_path(resolved))
    return str(p.parent / f'{p.stem}_fits.parquet')


def fits_path_ccs(resolved):
    """Derive the CCS fits parquet filename for a run."""
    p = Path(donut_path(resolved))
    return str(p.parent / f'{p.stem}_ccs_fits.parquet')


def plots_dir_ccs(resolved):
    """Derive the CCS plots output subdirectory for a run."""
    p = Path(donut_path(resolved))
    return str(p.parent / f'{p.stem}_ccs')


def log(msg):
    """Write to both stdout and log file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{timestamp}] {msg}'
    print(line)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def git_sync(message, extra_paths=None):
    """Stage runs.yaml + log files, commit, and push.

    Silent on failure — a broken git push shouldn't kill a pipeline run.
    Held under the same file lock as `update_run_status` so concurrent
    pipelines on the same machine serialize their git activity instead
    of fighting over the local index. Returns True if anything was
    committed, False otherwise.
    """
    paths = [str(RUNS_FILE), str(LOG_DIR)]
    if extra_paths:
        paths.extend(str(p) for p in extra_paths)
    try:
      with _file_lock(timeout_sec=60.0):
        # Stage the files (ok if some don't exist yet)
        subprocess.run(['git', 'add', '--'] + paths,
                       cwd=AOS_DIR.parent, check=False,
                       capture_output=True)
        # Check if there's anything staged
        result = subprocess.run(
            ['git', 'diff', '--cached', '--quiet'],
            cwd=AOS_DIR.parent, capture_output=True)
        if result.returncode == 0:
            return False  # nothing to commit
        subprocess.run(['git', 'commit', '-m', message],
                       cwd=AOS_DIR.parent, check=True,
                       capture_output=True)
        subprocess.run(['git', 'push'],
                       cwd=AOS_DIR.parent, check=False,
                       capture_output=True, timeout=30)
        return True
    except Exception as e:
        log(f'git_sync: skipped ({type(e).__name__}: {e})')
        return False


def build_command(run_name, step, resolved):
    """Build the shell command for a pipeline step."""
    dmin = resolved['day_obs_min']
    dmax = resolved['day_obs_max']
    coord = resolved.get('coord_sys', 'OCS')
    ps = resolved.get('param_set')

    if step == 'mktable':
        # Prefer --param-set if available, otherwise pass explicit args
        if ps:
            cmd = ['python', 'code/run_mktable.py',
                   '--param-set', str(ps),
                   '--day-obs-min', str(dmin),
                   '--day-obs-max', str(dmax)]
        else:
            cmd = ['python', 'code/run_mktable.py',
                   '--butler-repo', resolved['butler_repo'],
                   '--collections'] + resolved['fam_collections'] + [
                   '--programs'] + resolved['fam_programs'] + [
                   '--day-obs-min', str(dmin),
                   '--day-obs-max', str(dmax)]
        if resolved.get('calc_intrinsics'):
            cmd.append('--calc-intrinsics')
        for key in ['calc_mean_zernike', 'calc_focal_plane']:
            if resolved.get(key):
                cmd.append(f'--{key.replace("_", "-")}')
        if resolved.get('overwrite'):
            cmd.append('--overwrite')
        return cmd

    elif step == 'fit':
        h5 = hdf5_path(resolved)
        cmd = ['python', 'code/run_dz_fit.py', h5,
               '--coord-sys', coord]
        return cmd

    elif step == 'plots':
        # `plots` produces fit-parameter PDFs and trio comparisons only.
        # The slow per-image residual maps + movie now live in `movie`.
        h5 = hdf5_path(resolved)
        fp = fits_path(resolved)
        cmd = ['python', 'code/run_dz_plots.py', h5, fp,
               '--coord-sys', coord,
               '--no-single-image']
        for flag in ['no_fit_params', 'no_trio']:
            if resolved.get(flag):
                cmd.append(f'--{flag.replace("_", "-")}')
        return cmd

    elif step == 'movie':
        # `movie` produces only the per-image residual maps + ffmpeg movie.
        h5 = hdf5_path(resolved)
        fp = fits_path(resolved)
        cmd = ['python', 'code/run_dz_plots.py', h5, fp,
               '--coord-sys', coord,
               '--no-fit-params', '--no-trio']
        return cmd

    elif step == 'fit_ccs':
        h5 = hdf5_path(resolved)
        cmd = ['python', 'code/run_dz_fit.py', h5,
               '--coord-sys', 'CCS',
               '--output', fits_path_ccs(resolved)]
        return cmd

    elif step == 'plots_ccs':
        h5 = hdf5_path(resolved)
        fp = fits_path_ccs(resolved)
        cmd = ['python', 'code/run_dz_plots.py', h5, fp,
               '--coord-sys', 'CCS',
               '--output-dir', plots_dir_ccs(resolved),
               '--no-single-image']
        for flag in ['no_fit_params', 'no_trio']:
            if resolved.get(flag):
                cmd.append(f'--{flag.replace("_", "-")}')
        return cmd

    elif step == 'movie_ccs':
        h5 = hdf5_path(resolved)
        fp = fits_path_ccs(resolved)
        cmd = ['python', 'code/run_dz_plots.py', h5, fp,
               '--coord-sys', 'CCS',
               '--output-dir', plots_dir_ccs(resolved),
               '--no-fit-params', '--no-trio']
        return cmd

    else:
        raise ValueError(f'Unknown step: {step}')


def step_log_path(run_name, step):
    """Path to the per-step output log file."""
    return LOG_DIR / f'{run_name}_{step}.log'


def run_step(run_name, step, resolved, data, dry_run=False):
    """Execute one pipeline step, updating status in runs.yaml."""
    cmd = build_command(run_name, step, resolved)
    cmd_str = ' '.join(cmd)

    if dry_run:
        log(f'[DRY RUN] {run_name}.{step}: {cmd_str}')
        return True

    log(f'{run_name}.{step}: START — {cmd_str}')

    # Mark as running (atomic merge — safe when other pipelines are
    # updating different chunks in parallel).
    update_run_status(run_name, {step: 'running'})

    # Capture output to per-step log file while also printing to stdout
    step_log = step_log_path(run_name, step)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        with open(step_log, 'w') as f:
            f.write(f'# {run_name}.{step}\n')
            f.write(f'# {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'# {cmd_str}\n\n')
            f.flush()
            # Tee output to both file and stdout
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            proc.wait()

        elapsed = time.time() - t0
        elapsed_str = format_duration(elapsed)

        if proc.returncode == 0:
            update_run_status(run_name, {step: 'done'})
            log(f'{run_name}.{step}: DONE ({elapsed_str}) — log: {step_log}')
            git_sync(f'{run_name}.{step}: done ({elapsed_str})')
            return True
        else:
            update_run_status(run_name, {step: 'failed'})
            log(f'{run_name}.{step}: FAILED exit={proc.returncode} '
                f'({elapsed_str}) — log: {step_log}')
            git_sync(f'{run_name}.{step}: failed ({elapsed_str})')
            return False

    except Exception as e:
        elapsed = time.time() - t0
        update_run_status(run_name, {step: 'failed'})
        log(f'{run_name}.{step}: ERROR — {e} ({format_duration(elapsed)})')
        git_sync(f'{run_name}.{step}: error')
        return False


def format_duration(seconds):
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f'{seconds:.0f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.1f}m'
    else:
        return f'{seconds / 3600:.1f}h'


def cmd_status(data, run_filter=None):
    """Print status of all runs."""
    runs = data.get('runs', {})
    if not runs:
        print('No runs defined in runs.yaml')
        return

    param_sets = load_param_sets()

    # Header
    step_hdr = ' '.join(f'{s:>9s}' for s in STEP_ORDER)
    print(f'\n{"Run":<20s} {"param_set":<25s} {"days":<22s}  {step_hdr}')
    print('-' * (48 + 24 + len(step_hdr)))

    for name, cfg in runs.items():
        if run_filter and name != run_filter:
            continue
        resolved = resolve_run(cfg, param_sets)
        dmin = resolved.get('day_obs_min', '?')
        dmax = resolved.get('day_obs_max', '?')
        ps_name = cfg.get('param_set', '')
        steps = cfg.get('steps', {})

        statuses = [format_status(steps.get(s, 'pending')) for s in STEP_ORDER]
        step_row = ' '.join(f'{st:>9s}' for st in statuses)

        print(f'{name:<20s} {ps_name:<25s} {dmin}-{dmax}   {step_row}')

    # Summary
    total = len(runs)
    all_done = sum(1 for cfg in runs.values()
                   if all(cfg.get('steps', {}).get(s) == 'done'
                          for s in STEP_ORDER))
    print(f'\n{all_done}/{total} runs fully complete.\n')


def format_status(st):
    """Format status with indicators."""
    symbols = {
        'done': 'done',
        'running': 'RUN..',
        'pending': '---',
        'failed': 'FAIL',
        'skip': 'skip',
    }
    return symbols.get(st, st)


def _normalize_step_filter(step_filter):
    """Accept None, a single step name, or a list/tuple of step names.

    Returns a set or None.  Validates step names are known.
    """
    if step_filter is None:
        return None
    if isinstance(step_filter, str):
        step_filter = [step_filter]
    if not step_filter:
        return None
    unknown = [s for s in step_filter if s not in STEP_ORDER]
    if unknown:
        raise ValueError(
            f'Unknown step(s): {unknown}. Must be from {STEP_ORDER}')
    return set(step_filter)


def cmd_run(data, run_filter=None, step_filter=None, dry_run=False):
    """Run pending steps.

    step_filter may be None (run every pending step), a single step name,
    or a list of step names — for example `['fit', 'plots']` to run
    just those two together.
    """
    runs = data.get('runs', {})
    param_sets = load_param_sets()
    filter_set = _normalize_step_filter(step_filter)

    for name, cfg in runs.items():
        if run_filter and name != run_filter:
            continue

        resolved = resolve_run(cfg, param_sets)
        steps = cfg.get('steps', {})
        for step in STEP_ORDER:
            if filter_set is not None and step not in filter_set:
                continue

            status = steps.get(step, 'pending')
            if status not in ('pending', 'failed'):
                continue

            # Check prerequisites using dependency map.  When prereqs
            # come from another step that is *also* in this batch, treat
            # them as effectively satisfied (we just ran or are running
            # them now).
            prereqs_met = True
            for prev_step in STEP_DEPS.get(step, []):
                prev_status = steps.get(prev_step, 'pending')
                if prev_status != 'done':
                    if filter_set is not None and prev_step in filter_set:
                        # Will be / was just satisfied as part of this batch.
                        continue
                    if filter_set is not None:
                        log(f'{name}.{step}: prerequisite {prev_step} '
                            f'is {prev_status}, skipping')
                    prereqs_met = False
                    break

            if not prereqs_met:
                continue

            run_step(name, step, resolved, data, dry_run=dry_run)
            # Non-linear deps: don't break on failure; downstream prereqs
            # check handles it naturally (independent branches continue).


def cmd_set(data, run_name, step, status, cascade=True):
    """Manually set a step's status.

    When the new status is `pending` and `cascade=True` (default), every
    step that transitively depends on `step` (per STEP_DEPS) is also
    forced back to `pending`. That avoids leaving downstream steps
    marked `done` while their input has been invalidated — e.g.
    resetting mktable also resets fit/plots/movie/fit_ccs/plots_ccs/
    movie_ccs; resetting fit also resets plots/movie. Other status
    changes (`done`, `failed`, `skip`) don't cascade.
    """
    if run_name not in data.get('runs', {}):
        print(f'Unknown run: {run_name}')
        sys.exit(1)
    if step not in STEP_ORDER:
        print(f'Unknown step: {step}. Must be one of {STEP_ORDER}')
        sys.exit(1)
    if status not in VALID_STATUSES:
        print(f'Unknown status: {status}. Must be one of {VALID_STATUSES}')
        sys.exit(1)

    # Atomic per-run merge so concurrent pipelines on other chunks
    # aren't clobbered. update_run_status handles the cascade.
    pre = data['runs'][run_name].get('steps', {})
    post = update_run_status(run_name, {step: status}, cascade=cascade)
    cascaded = sorted(
        s for s in post
        if s != step and post.get(s) == 'pending' and pre.get(s) != 'pending')
    if cascaded:
        print(f'Set {run_name}.{step} = {status} '
              f'(also reset: {", ".join(cascaded)})')
    else:
        print(f'Set {run_name}.{step} = {status}')


def cmd_reset(data, run_name):
    """Reset all steps for a run to pending."""
    if run_name not in data.get('runs', {}):
        print(f'Unknown run: {run_name}')
        sys.exit(1)

    update_run_status(run_name, {step: 'pending' for step in STEP_ORDER})
    print(f'Reset all steps for {run_name} to pending')


def main():
    parser = argparse.ArgumentParser(
        description='AOS intrinsic wavefront analysis pipeline runner.')
    sub = parser.add_subparsers(dest='command')

    # status
    p_status = sub.add_parser('status', help='Show run status')
    p_status.add_argument('run', nargs='?', help='Filter to one run')

    # run
    p_run = sub.add_parser(
        'run', help='Execute pending steps',
        description=(
            'Run one or more steps for a chunk.  Pass multiple step names '
            'to execute them in sequence (in STEP_ORDER), e.g. '
            '`run chunk5 fit plots`.'))
    p_run.add_argument('run', nargs='?', help='Run name (default: all)')
    p_run.add_argument('step', nargs='*',
                       help='Step name(s); empty = all pending steps')
    p_run.add_argument('--dry-run', action='store_true',
                       help='Show commands without executing')

    # set
    p_set = sub.add_parser('set', help='Manually set step status')
    p_set.add_argument('run', help='Run name')
    p_set.add_argument('step', help='Step name')
    p_set.add_argument('status', help='New status')

    # reset
    p_reset = sub.add_parser('reset', help='Reset run to pending')
    p_reset.add_argument('run', help='Run name')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    data = load_runs()

    if args.command == 'status':
        cmd_status(data, args.run)
    elif args.command == 'run':
        # args.step is a list (possibly empty) thanks to nargs='*'.
        step_filter = args.step if args.step else None
        cmd_run(data, args.run, step_filter, dry_run=args.dry_run)
    elif args.command == 'set':
        cmd_set(data, args.run, args.step, args.status)
    elif args.command == 'reset':
        cmd_reset(data, args.run)


if __name__ == '__main__':
    main()
