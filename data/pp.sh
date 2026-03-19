awk '{
    qid = $1
    rel = $2
    nonrel = $3

    # Relevant qrels (label = 1)
    print qid "\tQ0\t" rel "\t1" >> "train_qrels_rel.txt"

    # Non-relevant qrels (label = 0)
    print qid "\tQ0\t" nonrel "\t0" >> "train_qrels_nonrel.txt"
}' qidpidtriples.train.top3.tsv
