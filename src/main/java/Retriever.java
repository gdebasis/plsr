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

    public List<MsMarcoQuery> loadQueriesAsList(String queryFile) throws Exception {
        BufferedReader br = new BufferedReader(new FileReader(queryFile));
        String line;
        List<MsMarcoQuery> qList = new ArrayList<>();

        while ((line=br.readLine())!=null) {
            String[] tokens = line.split("\t");
            qList.add(new MsMarcoQuery(tokens[0], tokens[1]));
        }
        br.close();

        return qList;
    }

    public Query makeQuery(MsMarcoQuery query) throws Exception {
        String queryText = query.qText;
        return makeQuery(queryText);
    }

    public Query makeQuery(String queryText) throws Exception {
        BooleanQuery.Builder qb = new BooleanQuery.Builder();
        String[] tokens = MsMarcoIndexer.analyze(MsMarcoIndexer.constructAnalyzer(), queryText).split("\\s+");
        for (String token: tokens) {
            TermQuery tq = new TermQuery(new Term(Constants.CONTENT_FIELD, token));
            qb.add(new BooleanClause(tq, BooleanClause.Occur.SHOULD));
        }
        return (Query)qb.build();
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

            MsMarcoQuery msMarcoQuery = new MsMarcoQuery(qid, queryText);
            Query luceneQuery = msMarcoQuery.getQuery();

            //System.out.println(String.format("Retrieving for query %s: %s", qid, luceneQuery));
            topDocs = searcher.search(luceneQuery, Constants.NUM_WANTED);
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
        if (args.length < 4) {
            args = new String[4];
            args[0] = Constants.MSMARCO_INDEX;
            args[1] = Constants.QRELS_TRAIN;
            args[2] = Constants.QRELS_DL19;
            args[3] = Constants.QUERIES_DL19;
        }

        Retriever retriever = new Retriever(args[0], args[3]);
        IndexUtils.init(retriever.searcher);
        RelEstimator relEstimator = new RelEstimator(retriever.reader, args[1], Constants.RELPRIOR_WEIGHTS_FILE);

        String[] similarityNames = { "LM-Dir", "LM-Dir-relp", "LM-Dir-allp"};
        Similarity[] sims = {
            //new BM25Similarity(),
            new LMDirichletSimilarity(1000),
            new TermPriorSimilarity(new LMDirichletSimilarity(1000), relEstimator.computeBetas())
        };

        AllRelRcds relRcds = new AllRelRcds(args[2]);

        for (int i=0; i < sims.length; i++) {
            Evaluator evaluator = new Evaluator(relRcds, retriever.retrieve(sims[i]));
            System.out.println(similarityNames[i]);
            System.out.println(evaluator.computeAll());
        }

        retriever.reader.close();
    }
}
