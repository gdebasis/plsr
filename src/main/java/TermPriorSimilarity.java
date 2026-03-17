import org.apache.lucene.util.BytesRef;
import java.util.Map;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.similarities.Similarity;

public class TermPriorSimilarity extends Similarity {

    private final Similarity baseSimilarity;
    private final Map<String, Double> beta;

    public TermPriorSimilarity(Similarity baseSimilarity,
                               Map<String, Double> beta) {
        this.baseSimilarity = baseSimilarity;
        this.beta = beta;
    }

    @Override
    public long computeNorm(org.apache.lucene.index.FieldInvertState state) {
        return baseSimilarity.computeNorm(state);
    }

    @Override
    public SimScorer scorer(float boost,
                            CollectionStatistics collectionStats,
                            TermStatistics... termStats) {

        SimScorer baseScorer =
                baseSimilarity.scorer(boost, collectionStats, termStats);

        float betaFactor = computeBetaFactor(termStats);

        return new SimScorer() {

            @Override
            public float score(float freq, long norm) {
                float baseScore = baseScorer.score(freq, norm);
                return baseScore * betaFactor;
            }

            @Override
            public Explanation explain(Explanation freq, long norm) {
                Explanation baseExpl =
                        baseScorer.explain(freq, norm);

                return Explanation.match(
                        baseExpl.getValue().floatValue() * betaFactor,
                        "beta-weighted score",
                        baseExpl
                );
            }
        };
    }

    private float computeBetaFactor(TermStatistics[] termStats) {

        if (termStats == null || termStats.length == 0)
            return 1.0f;

        double sum = 0.0;
        int count = 0;

        for (TermStatistics ts : termStats) {
            BytesRef br = ts.term();
            String term = br.utf8ToString();

            double b = beta.getOrDefault(term, 1.0);
            sum += b;
            count++;
        }

        return (float)(sum / count);
    }
}