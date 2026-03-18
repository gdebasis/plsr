import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.Similarity;

import java.io.File;
import java.util.Map;

public class RSJRetriever extends Retriever {
    RelEstimator relEstimator;
    Map<String, Double> termPriors;

    public RSJRetriever(String indexDir, String queryFile, String qrelsFile) throws Exception {
        super(indexDir, queryFile, new File(queryFile).getName() + ".res", "english");

        relEstimator = new RelEstimator(reader, qrelsFile, Constants.RELPRIOR_WEIGHTS_FILE);
        termPriors = relEstimator.computeBetas();
    }

    public Query makeQuery(String qid, String queryText) {
        BooleanQuery.Builder qb = new BooleanQuery.Builder();

        // Use your analyzer pipeline
        String analyzed = MsMarcoIndexer.analyze(MsMarcoIndexer.constructAnalyzer(), queryText);
        String[] tokens = analyzed.split("\\s+");

        for (String token : tokens) {
            if (token.isEmpty()) continue;

            double b = termPriors.getOrDefault(token, 0.0);

            // Option 1: skip non-informative terms (recommended)
            if (b <= 0) continue;

            TermQuery tq = new TermQuery(new Term(Constants.CONTENT_FIELD, token));

            BoostQuery boosted = new BoostQuery(tq, (float) b);
            qb.add(new BooleanClause(boosted, BooleanClause.Occur.SHOULD));
        }

        return qb.build();
    }

    public static void main(String[] args) throws Exception {
        final String indexDir = args.length < 4? Constants.MSMARCO_INDEX : args[0];
        final String qrelsTrain = args.length < 4? Constants.QRELS_TRAIN : args[1];
        final String qrelsTest = args.length < 4? Constants.QRELS_DL19 : args[2];
        final String queriesTest = args.length < 4? Constants.QUERIES_DL19 : args[3];

        Retriever retriever = new RSJRetriever(indexDir, queriesTest, qrelsTrain);
        AllRelRcds relRcds = new AllRelRcds(qrelsTest);

        Evaluator evaluator = new Evaluator(relRcds, retriever.retrieve(new LMDirichletSimilarity(1000)));
        System.out.println(evaluator.computeAll());

        retriever.reader.close();
    }
}
