#!/bin/tcsh
# Script to run variant analysis with the target output directory

# Set default parameters
set GROUP = 1
set OUTPUT_DIR = "/scratch/ml-csm/projects/fgenom/gve/output/kmeans/var_ana/"
set MODEL = "both"

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

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the python script
cd /home/sdodl001/UAVarPrior
python -m uavarprior.interpret.variant_analysis --group $GROUP --output-dir $OUTPUT_DIR --model $MODEL

echo "Analysis complete!"
