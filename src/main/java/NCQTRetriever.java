import org.apache.lucene.search.Query;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;

import java.io.File;

public class NCQTRetriever extends Retriever {
    QueryTermTranslator translator;
    QueryTermTranslator neg_translator;
    int numExpansionTerms;

    public NCQTRetriever(String indexDir, String testqueryFile,
                         String trainQrelsFilePos, String trainQrelsFileNeg,
                         String trainQueryFile, int numExpansionTerms) throws Exception {
        super(indexDir, testqueryFile, new File(testqueryFile).getName() + ".res", "english");

        this.numExpansionTerms = numExpansionTerms;

        System.out.println("Training from rel docs...");
        translator = new QueryTermTranslator(reader, Constants.CONTENT_FIELD, trainQrelsFilePos, trainQueryFile,
                1, 20, 50);
        translator.train(3);

        System.out.println("Training from nonrel docs...");
        neg_translator = new QueryTermTranslator(reader, Constants.CONTENT_FIELD, trainQrelsFileNeg, trainQueryFile, 0,
                20, 50);
        neg_translator.train(3);
    }

    public Query makeQuery(String qid, String queryText) {
        Query expQuery = QueryTermTranslator.expandQueryNC(
                new MsMarcoQuery(qid, queryText).makeQuery(),
                translator, neg_translator,
                numExpansionTerms);
        System.out.println(expQuery);
        return expQuery;
    }

    public static void main(String[] args) throws Exception {
        final String indexDir = args.length < 6? Constants.MSMARCO_INDEX : args[0];
        final String qrelsTrainPos = args.length < 6? Constants.QRELS_TRAIN_POS : args[1];
        final String qrelsTrainNeg = args.length < 6? Constants.QRELS_TRAIN_NEG : args[2];
        final String queriesTrain = args.length < 6? Constants.QUERY_FILE_TRAIN : args[3];
        final String qrelsTest = args.length < 6? Constants.QRELS_DL19 : args[4];
        final String queriesTest = args.length < 6? Constants.QUERIES_DL19 : args[5];

        Retriever retriever = new NCQTRetriever(indexDir,
                queriesTest, qrelsTrainPos, qrelsTrainNeg,
                queriesTrain, 25);

        AllRelRcds relRcds = new AllRelRcds(qrelsTest);
        Evaluator evaluator = new Evaluator(relRcds, retriever.retrieve(new LMDirichletSimilarity(50)));
        System.out.println(evaluator.computeAll());

        retriever.reader.close();
    }
}
