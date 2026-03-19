import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.*;
import org.apache.lucene.util.BytesRef;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class QueryTermTranslator {
    IndexReader reader;
    String field;
    String modelFileName;
    String trainQrelsFile;
    String trainQueryFile;
    AllRelRcds qrels;
    int targetLabel;

    // t(w | q)
    private Map<String, Map<String, Double>> t;
    LuceneDocTermProvider docProvider;

    private final int maxDocTerms = 100;   // prune doc terms
    private final int maxQueryTerms = 10; // safety
    private final int maxTranslations = 50; // prune per term

    Map<String, List<String>> queryTermsCache;
    Map<String, List<String>> docTermsCache;
    Map<String, String> queries;

    public QueryTermTranslator(IndexReader reader, String field, String trainQrelsFile,
                               String trainQueryFile, int targetLabel) {
        this.reader = reader;
        this.field = field;
        this.targetLabel = targetLabel;

        this.modelFileName = String.format("%s.cocc.%d.tsv", trainQrelsFile, targetLabel);
        this.trainQrelsFile = trainQrelsFile;
        this.trainQueryFile = trainQueryFile;

        t = new HashMap<>();
        docProvider = new VectorBasedDocTermProvider(reader, field);
    }

    // -------------------------------
    // Initialization
    // -------------------------------

    private void initialize() throws Exception {
        Map<String, Set<String>> cooc = new HashMap<>();

        System.out.println("Loading training qrels...");
        qrels = new AllRelRcds(trainQrelsFile, targetLabel);

        System.out.println("Loading training queries...");
        queries = IndexUtils.loadQueries(trainQueryFile);

        System.out.println("Analysing queries...");
        queryTermsCache = new HashMap<>();
        int queryCount = 0;
        for (Map.Entry<String, String> e : queries.entrySet()) {
            if (queryCount++%1000 == 0)
                System.out.print(String.format("Iterated over %d queries\r", queryCount));

            List<String> toks = tokenize(e.getValue());
            if (!toks.isEmpty()) {
                queryTermsCache.put(e.getKey(), toks);
            }
        }

        System.out.println("Analysing documents...");
        docTermsCache = new HashMap<>();

        int docCount = 0;
        for (PerQueryRelDocs perQuery : qrels.perQueryRels.values()) {
            for (String docId : perQuery.getRelDocs()) {
                if (docTermsCache.containsKey(docId)) continue;

                List<String> dTerms = docProvider.getDocTerms(docId, maxDocTerms);
                if (!dTerms.isEmpty()) {
                    docTermsCache.put(docId, dTerms);
                }

                if (docCount++%1000 == 0)
                    System.out.print(String.format("Iterated over %d documents\r", docCount));
            }
        }

        for (PerQueryRelDocs perQuery : qrels.perQueryRels.values()) {
            List<String> qTerms = queryTermsCache.get(perQuery.qid);
            if (qTerms == null) continue;

            for (String docId : perQuery.getRelDocs()) {
                List<String> dTerms = docTermsCache.get(docId);
                if (dTerms == null) continue;

                for (String q : qTerms) {
                    cooc.computeIfAbsent(q, k -> new HashSet<>())
                            .addAll(dTerms);
                }
            }
        }

        // uniform init
        for (String q : cooc.keySet()) {
            Set<String> ws = cooc.get(q);
            double p = 1.0 / ws.size();

            Map<String, Double> dist = new HashMap<>();
            for (String w : ws) dist.put(w, p);

            t.put(q, dist);
        }
    }

    // -------------------------------
    // 1. TRAINING
    // -------------------------------

    public void train(int numIters) throws Exception {
        File f = new File(modelFileName);
        // 1. If model exists → load and skip training
        if (f.exists() && f.length() > 0) {
            System.out.println("Loading translation model from " + modelFileName);
            load(modelFileName);
            return;
        }

        // Initialize uniformly
        initialize();

        System.out.println("Training word alignment model");

        for (int iter = 1; iter <= numIters; iter++) {
            System.out.print(String.format("EM Iteration: %d\n", iter));

            Map<String, Map<String, Double>> count = new HashMap<>();
            Map<String, Double> totalQ = new HashMap<>();

            int queryCount = 0;
            for (PerQueryRelDocs perQuery : qrels.perQueryRels.values()) {

                if (queryCount++%1000 == 0)
                    System.out.print(String.format("Iterated over %d queries\r", queryCount));

                String qid = perQuery.qid;
                List<String> qTerms = queryTermsCache.get(qid);
                if (qTerms == null) continue;

                for (String docId : perQuery.getRelDocs()) {
                    List<String> dTerms = docTermsCache.get(docId);
                    if (dTerms == null) continue;

                    for (String w : dTerms) {
                        double z = 0.0;
                        for (String q : qTerms) {
                            z += t.getOrDefault(q, Collections.emptyMap())
                                    .getOrDefault(w, 0.0);
                        }

                        if (z == 0) continue;

                        for (String q : qTerms) {
                            double val = t.getOrDefault(q, Collections.emptyMap())
                                    .getOrDefault(w, 0.0) / z;

                            count
                                    .computeIfAbsent(q, k -> new HashMap<>())
                                    .merge(w, val, Double::sum);

                            totalQ.merge(q, val, Double::sum);
                        }
                    }
                }
            }

            // M-step
            for (String q : count.keySet()) {
                Map<String, Double> dist = count.get(q);
                double total = totalQ.getOrDefault(q, 1.0);

                Map<String, Double> newDist = new HashMap<>();
                for (Map.Entry<String, Double> e : dist.entrySet()) {
                    newDist.put(e.getKey(), e.getValue() / total);
                }

                t.put(q, prune(newDist, maxTranslations));
            }
        }

        // 4. Save model
        System.out.println("Saving translation model to " + modelFileName);
        save(modelFileName);
    }

    // -------------------------------
    // Tokenization (reuse your analyzer ideally)
    // -------------------------------
    private List<String> tokenize(String text) {
        String analyzed = MsMarcoIndexer.analyze(MsMarcoIndexer.constructAnalyzer(), MsMarcoIndexer.normalizeNumbers(text));
        String[] toks = analyzed.split("\\s+");

        List<String> list = new ArrayList<>();
        for (String t : toks) {
            if (!t.isEmpty()) list.add(t);
        }
        return list.subList(0, Math.min(list.size(), maxQueryTerms));
    }

    // -------------------------------
    // Prune top-K
    // -------------------------------

    private Map<String, Double> prune(Map<String, Double> map, int k) {

        Map<String, Double> topK = map.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(k)
                .collect(HashMap::new,
                        (m, e) -> m.put(e.getKey(), e.getValue()),
                        HashMap::putAll);

        // Renormalize
        double sum = topK.values().stream().mapToDouble(Double::doubleValue).sum();

        if (sum > 0) {
            for (Map.Entry<String, Double> e : topK.entrySet()) {
                e.setValue(e.getValue() / sum);
            }
        }

        return topK;
    }

    /*
    private Map<String, Double> prune(Map<String, Double> map, int k) {
        return map.entrySet().stream()
            .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
            .limit(k)
            .collect(HashMap::new,
                    (m, e) -> m.put(e.getKey(), e.getValue()),
                    HashMap::putAll);
    }
    */

    // -------------------------------
    // 2. SAVE / LOAD
    // -------------------------------

    public void save(String file) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(file));

        for (String q : t.keySet()) {
            for (Map.Entry<String, Double> e : t.get(q).entrySet()) {
                bw.write(q + "\t" + e.getKey() + "\t" + e.getValue());
                bw.newLine();
            }
        }
        bw.close();
    }

    public void load(String file) throws IOException {
        t.clear();

        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;

        while ((line = br.readLine()) != null) {
            String[] parts = line.split("\t");
            if (parts.length != 3) continue;

            String q = parts[0];
            String w = parts[1];
            double val = Double.parseDouble(parts[2]);

            t.computeIfAbsent(q, k -> new HashMap<>())
                    .put(w, val);
        }
        br.close();
    }

    // -------------------------------
    // 3. QUERY EXPANSION
    // -------------------------------

    static public Query expandQuery(Query originalQuery, QueryTermTranslator translator, int m) {
        BooleanQuery.Builder qb = new BooleanQuery.Builder();
        if (!(originalQuery instanceof BooleanQuery)) {
            return originalQuery;
        }

        BooleanQuery bq = (BooleanQuery) originalQuery;
        for (BooleanClause clause : bq.clauses()) {
            Query q = clause.getQuery();
            if (!(q instanceof TermQuery)) continue;

            TermQuery tq = (TermQuery) q;
            qb.add(new BoostQuery(tq, 1.0f), BooleanClause.Occur.SHOULD); // the original query term with weight 1

            String term = tq.getTerm().text();

            Map<String, Double> expansions = translator.t.get(term);
            if (expansions == null) continue;

            int count = 0;
            for (Map.Entry<String, Double> e :
                    expansions.entrySet().stream()
                    .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                    .collect(Collectors.toList())) {

                if (count++ >= m) break;
                if (e.getKey().equals(term)) continue;

                TermQuery newTq = new TermQuery(
                        new Term(Constants.CONTENT_FIELD, e.getKey()));

                qb.add(new BoostQuery(newTq, e.getValue().floatValue()), BooleanClause.Occur.SHOULD);
            }
        }

        return qb.build();
    }

    public Map<String, Double> getTranslations(String term) {
        return t.get(term);
    }

    // -------------------------------
    // Interface for doc term access
    // -------------------------------

    public interface LuceneDocTermProvider {
        List<String> getDocTerms(String docId, int maxTerms);
    }

    public static Query expandQueryNC(
            Query originalQuery,
            QueryTermTranslator relModel,
            QueryTermTranslator nonRelModel,
            int m) {

        BooleanQuery.Builder qb = new BooleanQuery.Builder();
        if (!(originalQuery instanceof BooleanQuery)) {
            return originalQuery;
        }

        BooleanQuery bq = (BooleanQuery) originalQuery;
        final double EPS = 1e-9;

        for (BooleanClause clause : bq.clauses()) {
            Query q = clause.getQuery();
            if (!(q instanceof TermQuery)) continue;

            TermQuery tq = (TermQuery) q;
            String term = tq.getTerm().text();

            // Keep original term (anchor)
            qb.add(new BoostQuery(tq, 1.0f), BooleanClause.Occur.SHOULD);

            Map<String, Double> tr = relModel.getTranslations(term);
            Map<String, Double> tn = nonRelModel.getTranslations(term);

            if (tr == null) continue;

            // Collect candidate weights
            List<Map.Entry<String, Double>> weighted = new ArrayList<>();

            for (Map.Entry<String, Double> e : tr.entrySet()) {
                String w = e.getKey();
                double pr = e.getValue();

                if (w.equals(term)) continue;

                double pn = (tn == null) ? EPS : tn.getOrDefault(w, EPS);

                // KL contribution
                //double weight = pr * Math.log((pr + EPS) / (pn + EPS));
                double weight = Math.log(1+ (pr + EPS) / (pn + EPS));

                // Optional: skip negative contributions
                if (weight <= 0) continue;

                weighted.add(new AbstractMap.SimpleEntry<>(w, weight));
            }

            // sort descending
            weighted.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
            int count = 0;

            for (Map.Entry<String, Double> e : weighted) {
                if (count++ >= m) break;

                TermQuery newTq = new TermQuery(new Term(Constants.CONTENT_FIELD, e.getKey()));
                qb.add(new BoostQuery(newTq, e.getValue().floatValue()), BooleanClause.Occur.SHOULD);
            }
        }

        return qb.build();
    }
}

class VectorBasedDocTermProvider implements QueryTermTranslator.LuceneDocTermProvider {

    private final IndexReader reader;
    private final String field;

    public VectorBasedDocTermProvider(IndexReader reader, String field) {
        this.reader = reader;
        this.field = field;
    }

    @Override
    public List<String> getDocTerms(String docId, int maxTerms) {
        try {
            int docOffset = IndexUtils.getDocOffsetFromId(docId);
            if (docOffset < 0) return Collections.emptyList();

            Terms terms = reader.getTermVector(docOffset, field);
            if (terms == null) return Collections.emptyList();

            TermsEnum te = terms.iterator();
            BytesRef term;

            List<String> result = new ArrayList<>();

            while ((term = te.next()) != null) {
                result.add(term.utf8ToString());

                if (result.size() >= maxTerms) break; // early stop
            }

            return result;

        } catch (IOException e) {
            e.printStackTrace();
            return Collections.emptyList();
        }
    }
}

