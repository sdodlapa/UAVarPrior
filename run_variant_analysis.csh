#!/bin/tcsh
# 
# Script to run variant analysis with the target output directory
# 
# Created by: Sanjeeva Reddy Dodlapati
# Date: May 22, 2025
#
# Set default parameters
set GROUP = 1
set OUTPUT_DIR = "/scratch/ml-csm/projects/fgenom/gve/output/kmeans/outputs/"
set MODEL = "both"
set REBUILD_MATRIX = 0  # 0 means use cached matrix if available

# Parse command-line arguments
while ( $#argv > 0 )
    switch ( $argv[1] )
        case "--group":
            shift
            set GROUP = $argv[1]
            shift
            breaksw
        case "--output-dir":
            shift
            set OUTPUT_DIR = $argv[1]
            shift
            breaksw
        case "--model":
            shift
            set MODEL = $argv[1]
            shift
            breaksw
        case "--rebuild-matrix":
            set REBUILD_MATRIX = 1  # Force rebuild of membership matrix
            shift
            breaksw
        case "--help":
            echo "Usage: run_variant_analysis.csh [options]"
            echo "Options:"
            echo "  --group N          Set group number (default: 1)"
            echo "  --output-dir DIR   Set output directory"
            echo "  --model TYPE       Set model type: pred1, pred150, both (default: both)"
            echo "  --rebuild-matrix   Force rebuild of the membership matrix"
            echo "  --help             Display this help message"
            exit 0
        default:
            echo "Unknown option: $argv[1]"
            shift
            breaksw
    endsw
end

# Print parameters
echo "Running variant analysis with the following parameters:"
echo "Group: $GROUP"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
if ($REBUILD_MATRIX == 1) then
    echo "Matrix caching: DISABLED (will rebuild matrix)"
else
    echo "Matrix caching: ENABLED (will use cached matrix if available)"
endif

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the python script
cd /home/sdodl001/UAVarPrior

# Build command with appropriate flags
set CMD = "python -m uavarprior.interpret.variant_analysis --group $GROUP --output-dir $OUTPUT_DIR --model $MODEL"

# Add rebuild-matrix flag if needed
if ($REBUILD_MATRIX == 1) then
    set CMD = "$CMD --rebuild-matrix"
endif

# Execute the command
echo "Running: $CMD"
$CMD

echo "Analysis complete!"
