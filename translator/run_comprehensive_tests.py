#!/usr/bin/env python3
"""
Comprehensive test runner for the FormalVerifML framework.
This script runs all tests including HuggingFace model testing and Lean verification.
"""

import subprocess
import json
import time
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any


class ComprehensiveTestRunner:
    """Runner for comprehensive testing of the FormalVerifML framework."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.start_time = time.time()

    def run_huggingface_tests(self) -> Dict[str, Any]:
        """Run HuggingFace model tests."""
        print("=" * 60)
        print("RUNNING HUGGINGFACE MODEL TESTS")
        print("=" * 60)

        try:
            cmd = [
                "py",
                "translator/test_huggingface_models.py",
                "--model_size",
                self.config.get("huggingface_model_size", "small"),
                "--output",
                "test_results_huggingface.json",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                with open("test_results_huggingface.json", "r") as f:
                    results = json.load(f)
                print("✅ HuggingFace tests completed successfully")
                return {"status": "success", "results": results}
            else:
                print(f"❌ HuggingFace tests failed: {result.stderr}")
                return {"status": "error", "error": result.stderr}

        except subprocess.TimeoutExpired:
            print("❌ HuggingFace tests timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ HuggingFace tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_lean_build(self) -> Dict[str, Any]:
        """Run Lean build to ensure all modules compile."""
        print("=" * 60)
        print("RUNNING LEAN BUILD")
        print("=" * 60)

        try:
            cmd = ["lake", "build"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print("✅ Lean build completed successfully")
                return {"status": "success"}
            else:
                print(f"❌ Lean build failed: {result.stderr}")
                return {"status": "error", "error": result.stderr}

        except subprocess.TimeoutExpired:
            print("❌ Lean build timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Lean build error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_lean_tests(self) -> Dict[str, Any]:
        """Run Lean verification tests."""
        print("=" * 60)
        print("RUNNING LEAN VERIFICATION TESTS")
        print("=" * 60)

        try:
            # Run Lean with our test suite
            cmd = ["lake", "exe", "FormalVerifML"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

            if result.returncode == 0:
                print("✅ Lean verification tests completed successfully")
                return {"status": "success", "output": result.stdout}
            else:
                print(f"❌ Lean verification tests failed: {result.stderr}")
                return {"status": "error", "error": result.stderr}

        except subprocess.TimeoutExpired:
            print("❌ Lean verification tests timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Lean verification tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_memory_tests(self) -> Dict[str, Any]:
        """Run memory optimization tests."""
        print("=" * 60)
        print("RUNNING MEMORY OPTIMIZATION TESTS")
        print("=" * 60)

        try:
            # Test memory-optimized transformer generation
            cmd = [
                "py",
                "translator/generate_lean_model.py",
                "--model_json",
                "translator/sample_transformer.json",
                "--output_lean",
                "lean/FormalVerifML/generated/memory_optimized_test.lean",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print("✅ Memory optimization tests completed successfully")
                return {"status": "success"}
            else:
                print(f"❌ Memory optimization tests failed: {result.stderr}")
                return {"status": "error", "error": result.stderr}

        except subprocess.TimeoutExpired:
            print("❌ Memory optimization tests timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ Memory optimization tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_smt_tests(self) -> Dict[str, Any]:
        """Run SMT solver integration tests."""
        print("=" * 60)
        print("RUNNING SMT SOLVER INTEGRATION TESTS")
        print("=" * 60)

        try:
            # Test SMT formula generation
            cmd = ["lake", "exe", "FormalVerifML"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print("✅ SMT integration tests completed successfully")
                return {"status": "success", "output": result.stdout}
            else:
                print(f"❌ SMT integration tests failed: {result.stderr}")
                return {"status": "error", "error": result.stderr}

        except subprocess.TimeoutExpired:
            print("❌ SMT integration tests timed out")
            return {"status": "timeout"}
        except Exception as e:
            print(f"❌ SMT integration tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_web_interface_tests(self) -> Dict[str, Any]:
        """Run web interface tests."""
        print("=" * 60)
        print("RUNNING WEB INTERFACE TESTS")
        print("=" * 60)

        try:
            # Start web server
            web_process = subprocess.Popen(
                ["py", "webapp/app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait for server to start
            time.sleep(5)

            # Test web interface
            import requests

            try:
                response = requests.get("http://localhost:5000", timeout=10)
                if response.status_code == 200:
                    print("✅ Web interface tests completed successfully")
                    result = {"status": "success", "status_code": response.status_code}
                else:
                    print(
                        f"❌ Web interface returned status code: {response.status_code}"
                    )
                    result = {"status": "error", "status_code": response.status_code}
            except requests.RequestException as e:
                print(f"❌ Web interface test failed: {str(e)}")
                result = {"status": "error", "error": str(e)}

            # Cleanup
            web_process.terminate()
            web_process.wait()

            return result

        except Exception as e:
            print(f"❌ Web interface tests error: {str(e)}")
            return {"status": "error", "error": str(e)}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("STARTING COMPREHENSIVE TEST SUITE")
        print("=" * 60)

        # Run all test categories
        test_categories = [
            ("lean_build", self.run_lean_build),
            ("huggingface_tests", self.run_huggingface_tests),
            ("lean_tests", self.run_lean_tests),
            ("memory_tests", self.run_memory_tests),
            ("smt_tests", self.run_smt_tests),
            ("web_interface_tests", self.run_web_interface_tests),
        ]

        for category, test_func in test_categories:
            if self.config.get(f"run_{category}", True):
                self.results[category] = test_func()
            else:
                print(f"Skipping {category} (disabled in config)")
                self.results[category] = {"status": "skipped"}

        return self.results

    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time

        # Count results
        total_tests = len(self.results)
        successful_tests = sum(
            1 for r in self.results.values() if r.get("status") == "success"
        )
        failed_tests = sum(
            1 for r in self.results.values() if r.get("status") == "error"
        )
        timeout_tests = sum(
            1 for r in self.results.values() if r.get("status") == "timeout"
        )
        skipped_tests = sum(
            1 for r in self.results.values() if r.get("status") == "skipped"
        )

        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

        report = f"""
COMPREHENSIVE TEST REPORT
{'='*60}
Total Test Categories: {total_tests}
Successful: {successful_tests}
Failed: {failed_tests}
Timeout: {timeout_tests}
Skipped: {skipped_tests}
Total Execution Time: {total_time:.2f}s
Success Rate: {success_rate:.1f}%

DETAILED RESULTS:
{'='*60}
"""

        for category, result in self.results.items():
            status = result.get("status", "unknown")
            report += f"\n{category.upper()}:\n"
            report += f"  Status: {status}\n"

            if "error" in result:
                report += f"  Error: {result['error']}\n"
            if "output" in result:
                report += f"  Output: {result['output'][:200]}...\n"

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for FormalVerifML"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="test_config.json",
        help="Path to test configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comprehensive_test_report.txt",
        help="Path to output report file",
    )
    parser.add_argument(
        "--huggingface-size",
        choices=["small", "medium", "all"],
        default="small",
        help="Size of HuggingFace models to test",
    )

    args = parser.parse_args()

    # Default configuration
    config = {
        "huggingface_model_size": args.huggingface_size,
        "run_lean_build": True,
        "run_huggingface_tests": True,
        "run_lean_tests": True,
        "run_memory_tests": True,
        "run_smt_tests": True,
        "run_web_interface_tests": True,
    }

    # Load custom config if provided
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            custom_config = json.load(f)
            config.update(custom_config)

    # Run tests
    runner = ComprehensiveTestRunner(config)
    results = runner.run_all_tests()

    # Generate and save report
    report = runner.generate_report()

    with open(args.output, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nDetailed report saved to: {args.output}")

    # Exit with appropriate code
    successful_tests = sum(1 for r in results.values() if r.get("status") == "success")
    total_tests = len(results)

    if successful_tests == total_tests:
        print("🎉 All tests passed!")
        exit(0)
    else:
        print(f"⚠️  {total_tests - successful_tests} test(s) failed")
        exit(1)


if __name__ == "__main__":
    main()
