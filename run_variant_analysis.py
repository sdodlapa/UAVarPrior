#!/usr/bin/env python3
"""
Script to run the full variant analysis for both pred1 and pred150 models.

This script runs the comprehensive variant analysis pipeline that includes:
1. Collecting unique variants
2. Building membership matrices
3. Analyzing cell-specific and cell-nonspecific variants
4. Computing profile similarity matrices
5. Saving all results and statistics

Usage:
    python run_variant_analysis.py --group 1 --model both
    python run_variant_analysis.py --group 1 --model pred1
    python run_variant_analysis.py --group 1 --model pred150
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the variant analysis module
try:
    from uavarprior.interpret.variant_analysis import run_full_analysis
    print("✅ Successfully imported variant analysis module")
except ImportError as e:
    print(f"❌ Failed to import variant analysis module: {e}")
    print("Make sure you're running from the UAVarPrior root directory")
    sys.exit(1)


def main():
    """Main function to run the variant analysis."""
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Run full variant analysis for pred1 and/or pred150 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run analysis for both models, group 1
    python run_variant_analysis.py --group 1 --model both
    
    # Run analysis for only pred1 model
    python run_variant_analysis.py --group 1 --model pred1
    
    # Run analysis for only pred150 model with custom output directory
    python run_variant_analysis.py --group 1 --model pred150 --output-dir /custom/path
    
    # Force rebuild of membership matrices
    python run_variant_analysis.py --group 1 --model both --rebuild-matrix
        """
    )
    
    parser.add_argument("--group", type=int, default=1,
                        help="Group number (default: 1)")
    parser.add_argument("--output-dir", type=str, 
                        default="/scratch/ml-csm/projects/fgenom/gve/output/kmeans/var_ana/",
                        help="Output directory for saving results")
    parser.add_argument("--model", type=str, choices=["pred1", "pred150", "both"], default="both",
                       help="Which prediction model to process (default: both)")
    parser.add_argument("--rebuild-matrix", action="store_true",
                        help="Always rebuild the membership matrix even if a cached one exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually running the analysis")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("🧬 UAVarPrior Variant Analysis Pipeline")
    print("=" * 80)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Group: {args.group}")
    print(f"🤖 Model(s): {args.model}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔄 Matrix caching: {'DISABLED (rebuild)' if args.rebuild_matrix else 'ENABLED'}")
    print("=" * 80)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No actual analysis will be performed")
        print("\nAnalysis would be performed with the following configuration:")
        print(f"  - run_full_analysis(")
        print(f"      group={args.group},")
        print(f"      save_results=True,")
        print(f"      output_dir='{args.output_dir}',")
        print(f"      model='{args.model}',")
        print(f"      use_cached_matrix={not args.rebuild_matrix}")
        print(f"    )")
        print("\nTo run the actual analysis, remove the --dry-run flag.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"📁 Output directory created/verified: {args.output_dir}")
    
    # Record start time
    start_time = time.time()
    
    try:
        print("\n🚀 Starting variant analysis...")
        print("-" * 40)
        
        # Run the analysis with the specified parameters
        results = run_full_analysis(
            group=args.group,
            save_results=True,
            output_dir=args.output_dir,
            model=args.model,
            use_cached_matrix=not args.rebuild_matrix
        )
        
        # Calculate runtime
        end_time = time.time()
        runtime = end_time - start_time
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"⏱️  Total runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        print(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Print summary of results
        print("\n📊 RESULTS SUMMARY:")
        print("-" * 40)
        
        if 'pred1' in results and results['pred1']['cell_specific'] is not None:
            print(f"🔬 pred1 model:")
            print(f"   - Cell-specific variants: {len(results['pred1']['cell_specific']):,}")
            print(f"   - Cell-nonspecific variants: {len(results['pred1']['cell_nonspecific']):,}")
        
        if 'pred150' in results and results['pred150']['cell_specific'] is not None:
            print(f"🔬 pred150 model:")
            print(f"   - Cell-specific variants: {len(results['pred150']['cell_specific']):,}")
            print(f"   - Cell-nonspecific variants: {len(results['pred150']['cell_nonspecific']):,}")
        
        if 'maf_data' in results:
            print(f"🧬 MAF data: {len(results['maf_data']):,} variants loaded")
        
        print(f"\n📁 All results saved to: {args.output_dir}")
        
        # List generated files
        print("\n📄 Generated files:")
        print("-" * 40)
        try:
            for file in sorted(os.listdir(args.output_dir)):
                if any(file.endswith(ext) for ext in ['.pkl', '.csv', '.npz', '.parquet', '.txt']):
                    file_path = os.path.join(args.output_dir, file)
                    size_mb = os.path.getsize(file_path) / 1024 / 1024
                    print(f"   📄 {file} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"   ⚠️  Could not list files: {e}")
        
        print("\n🎉 Analysis pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user (Ctrl+C)")
        print("Partial results may be saved in the output directory.")
        sys.exit(1)
        
    except Exception as e:
        end_time = time.time()
        runtime = end_time - start_time
        
        print("\n" + "=" * 80)
        print("❌ ANALYSIS FAILED!")
        print("=" * 80)
        print(f"⏱️  Runtime before failure: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        print(f"❌ Error: {str(e)}")
        
        if args.verbose:
            import traceback
            print("\n🔍 Detailed error traceback:")
            print("-" * 40)
            traceback.print_exc()
        
        print(f"\n💡 Troubleshooting tips:")
        print("   1. Check that all required data files exist")
        print("   2. Ensure sufficient disk space in output directory")
        print("   3. Verify Python environment has all required packages")
        print("   4. Try running with --rebuild-matrix flag")
        print("   5. Run with --verbose flag for more detailed error information")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
