#!/bin/bash

if [ $# -lt 3 ]
then
    echo "Usage: $0 <baseline res> <new_method res> <qrels>"
    exit 1
fi

BASELINE_RES=$1
SAMPLE_RES=$2
QRELS=$3
TRECEVAL="/Users/debasis/devtools/trec_eval/trec_eval -l2 -m all_trec" 

# -------------------------------
# Function to extract metric values
# -------------------------------
extract_metric() {
    local metric=$1
    local input=$2
    local output=$3
    local extra_flag=$4

    $TRECEVAL $extra_flag -q "$QRELS" "$input" \
    | awk -v m="$metric" '$1==m && $2!="all" {print $NF}' \
    > "$output"
}

# -------------------------------
# Write Python test once
# -------------------------------
cat > test.py << 'EOF'
from scipy import stats
import sys

f1, f2 = sys.argv[1], sys.argv[2]

with open(f1) as file:
    baseline = [float(x.strip()) for x in file]

with open(f2) as file:
    method = [float(x.strip()) for x in file]

print(stats.ttest_rel(baseline, method).pvalue)
EOF

# -------------------------------
# MAP
# -------------------------------
extract_metric "map" "$BASELINE_RES" map_b "-l2"
extract_metric "map" "$SAMPLE_RES" map_m "-l2"

python test.py map_b map_m

# -------------------------------
# NDCG@10
# -------------------------------
extract_metric "ndcg_cut_10" "$BASELINE_RES" ndcg10_b ""
extract_metric "ndcg_cut_10" "$SAMPLE_RES" ndcg10_m ""

python test.py ndcg10_b ndcg10_m

# -------------------------------
# num_rel_ret
# -------------------------------
extract_metric "num_rel_ret" "$BASELINE_RES" rb "-l2"
extract_metric "num_rel_ret" "$SAMPLE_RES" rm "-l2"

python test.py rb rm
