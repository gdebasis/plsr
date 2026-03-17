public interface Constants {
    boolean NORMALISE_NUMBERS = false;
    String ID_FIELD = "id";
    String CONTENT_FIELD = "words";

    String MSMARCO_COLL = "data/collection.tsv";
    String MSMARCO_INDEX = "index/";
    String MSMARCO_QUERY_INDEX = "query_index/";
    String QRELS_TRAIN = "data/qrels.train.tsv";
    String QUERY_FILE_TRAIN = "data/queries.train.tsv";
    String STOP_FILE = "stop.txt";
    String QRELS_TEST = "data/trecdl/trecdl1920.qrels";
    String RES_FILE = "ColBERT-PRF-VirtualAppendix/BM25/BM25.2019.res";
    int NUM_WANTED = 100;
    int EVAL_MIN_REL = 2;

    String QRELS_DL19 = "data/trecdl/pass_2019.qrels";
    String QRELS_DL20 = "data/trecdl/pass_2020.qrels";

    String QUERIES_DL19 = "data/trecdl/pass_2019.queries";
    String QUERIES_DL20 = "data/trecdl/pass_2020.queries";

    boolean AUTO_SORT_TOP_DOCS = true;
    String RELPRIOR_WEIGHTS_FILE = "relprior_weights.tsv";
}
