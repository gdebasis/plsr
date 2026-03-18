import org.apache.lucene.search.Query;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;

import java.io.File;

public class QTRetriever extends Retriever {
    QueryTermTranslator translator;
    int numExpansionTerms;

    public QTRetriever(String indexDir, String testqueryFile, String trainQrelsFile, String trainQueryFile, int numExpansionTerms) throws Exception {
        super(indexDir, testqueryFile, new File(testqueryFile).getName() + ".res", "english");

        AllRelRcds relRcds = new AllRelRcds(trainQrelsFile);
        this.numExpansionTerms = numExpansionTerms;
        translator = new QueryTermTranslator(reader, Constants.CONTENT_FIELD, trainQrelsFile, trainQueryFile, trainQrelsFile + ".cocc.tsv");
        translator.train(10);
    }

    public Query makeQuery(String qid, String queryText) {
        Query expQuery = translator.expandQuery(new MsMarcoQuery(qid, queryText).makeQuery(), numExpansionTerms);
        System.out.println(expQuery);
        return expQuery;
    }

    public static void main(String[] args) throws Exception {
        final String indexDir = args.length < 5? Constants.MSMARCO_INDEX : args[0];
        final String qrelsTrain = args.length < 5? Constants.QRELS_TRAIN : args[1];
        final String queriesTrain = args.length < 5? Constants.QUERY_FILE_TRAIN : args[2];
        final String qrelsTest = args.length < 5? Constants.QRELS_DL19 : args[3];
        final String queriesTest = args.length < 5? Constants.QUERIES_DL19 : args[4];

        Retriever retriever = new QTRetriever(indexDir, queriesTest, qrelsTrain, queriesTrain, 3);

        AllRelRcds relRcds = new AllRelRcds(qrelsTest);

        Evaluator evaluator = new Evaluator(relRcds, retriever.retrieve(new LMDirichletSimilarity(1000)));
        System.out.println(evaluator.computeAll());

        retriever.reader.close();
    }

}
