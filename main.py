
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path


def run_script(script_name: str):
    """Запустить Python скрипт как подпроцесс."""
    script_path = Path(__file__).parent / script_name
    result = subprocess.run([sys.executable, str(script_path)], check=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Pipeline orchestrator')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4],
                        help='Run only specific step')
    parser.add_argument('--skip', type=int, nargs='+', default=[],
                        help='Steps to skip (1-4)')
    args = parser.parse_args()

    steps = {
        1: 'step1_merge.py',
        2: '2_step_transform.py',  # includes step 3 (price tracking)
        4: '3_step_analytics.py',
    }

    print("=" * 60)
    print(f"🚀 PIPELINE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start = datetime.now()

    try:
        if args.step:
            # Один конкретный шаг
            if args.step in [1]:
                run_script('step1_merge.py')
            elif args.step in [2, 3]:
                run_script('step2_transform.py')
            elif args.step == 4:
                run_script('step3_analytics.py')
        else:
            # Полный пайплайн
            if 1 not in args.skip:
                print("\n>>> Running Step 1: Merge")
                run_script('step1_merge.py')
            else:
                print("Step 1 skipped")

            if 2 not in args.skip and 3 not in args.skip:
                print("\n>>> Running Step 2+3: Transform & Price Tracking")
                run_script('2_step_transform.py')
            else:
                print("Steps 2-3 skipped")

            if 4 not in args.skip:
                print("\n>>> Running Step 4: Analytics")
                run_script('3_step_analytics.py')
            else:
                print("Step 4 skipped")

        elapsed = (datetime.now() - start).total_seconds()
        print("=" * 60)
        print(f"✅ PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print("=" * 60)

    except subprocess.CalledProcessError as e:
        print(f"❌ PIPELINE FAILED: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
