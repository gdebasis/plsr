import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.search.similarities.LMJelinekMercerSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;


public class Retriever {
    IndexReader reader;
    IndexSearcher searcher;
    Similarity sim;
    String queryFile;
    Map<String, String> queries;
    List<MsMarcoQuery> queryList;
    String resFile;
    String analyzerName;

    public Retriever(String indexDir,
                     String queryFile,
                     String resFile,
                     String analyzerName) throws Exception {
        this.analyzerName = analyzerName;
        reader = DirectoryReader.open(FSDirectory.open(new File(Constants.MSMARCO_INDEX).toPath()));
        searcher = new IndexSearcher(reader);
        this.queryFile = queryFile;
        queries = IndexUtils.loadQueries(queryFile);
        queryList = new ArrayList<>();
        for (Map.Entry<String, String> e: queries.entrySet()) {
            queryList.add(new MsMarcoQuery(e.getKey(), e.getValue()));
        }
        this.resFile = resFile;
        IndexUtils.init(searcher);
    }

    public Retriever(String queryFile, String resFile, String analyzerName) throws Exception {
        this(Constants.MSMARCO_INDEX, queryFile, resFile, analyzerName);
    }

    public Retriever(String indexDir, String queryFile) throws Exception {
        this(indexDir, queryFile, new File(queryFile).getName() + ".res", "english");
    }

    public Retriever(String queryFile) throws Exception {
        this(Constants.MSMARCO_INDEX, queryFile, new File(queryFile).getName() + ".res", "english");
    }

    public Query makeQuery(String qid, String queryText) {
        return new MsMarcoQuery(qid, queryText).makeQuery();
    }

    public Map<String, TopDocs> retrieve(Similarity sim) throws Exception {
        searcher.setSimilarity(sim);

        Map<String, TopDocs> results = new HashMap<>();
        Map<String, String> testQueries = IndexUtils.loadQueries(queryFile);
        testQueries
            .entrySet()
            .stream()
            .collect(
                Collectors.toMap(
                e -> e.getKey(),
                e -> MsMarcoIndexer.normalizeNumbers(e.getValue()
                )
                )
            )
        ;

        System.out.println("Saving results in " + resFile);

        BufferedWriter bw = new BufferedWriter(new FileWriter(resFile));
        TopDocs topDocs;
        for (Map.Entry<String, String> e : testQueries.entrySet()) {
            String qid = e.getKey();
            String queryText = e.getValue();

            // Build weighted query
            Query weightedQuery = makeQuery(qid, queryText);
            topDocs = searcher.search(weightedQuery, Constants.NUM_WANTED);

            results.put(qid, topDocs);
            saveTopDocsResFile(bw, qid, queryText, topDocs);
        }
        bw.close();
        return results;
    }

    void saveTopDocsResFile(BufferedWriter bw, String qid, String queryText, TopDocs topDocs) throws Exception {
        int rank = 1;
        for (ScoreDoc sd: topDocs.scoreDocs) {
            int docId = sd.doc;
            bw.write(String.format("%s\tQ0\t%s\t%d\t%.4f\tthis_run\n", qid, reader.document(docId).get(Constants.ID_FIELD), rank++, sd.score));
        }
    }

    public List<MsMarcoQuery> getQueryList() { return queryList; }

    public static void main(String[] args) throws Exception {
        final String indexDir = args.length < 3? Constants.MSMARCO_INDEX : args[0];
        final String qrelsTest = args.length < 3? Constants.QRELS_DL19 : args[1];
        final String queriesTest = args.length < 3? Constants.QUERIES_DL19 : args[2];

        Retriever retriever = new Retriever(indexDir, queriesTest);
        AllRelRcds relRcds = new AllRelRcds(qrelsTest);

        Evaluator evaluator = new Evaluator(relRcds, retriever.retrieve(new LMDirichletSimilarity(500)));
        System.out.println(evaluator.computeAll());

        retriever.reader.close();
    }
}
