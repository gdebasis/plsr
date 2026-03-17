import org.apache.lucene.document.Document;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;

import java.util.List;
import java.util.Map;

public class Evaluator {
    AllRelRcds relRcds;
    AllRetrievedResults retRcds;

    public Evaluator(String resFile) {
        retRcds = new AllRetrievedResults(resFile);
    }

    public Evaluator(AllRelRcds relRcds, String resFile) {
        retRcds = new AllRetrievedResults(resFile);
        this.relRcds = relRcds;
    }

    public Evaluator(String qrelsFile, AllRetrievedResults allRetrievedResults) {
        relRcds = new AllRelRcds(qrelsFile);
        retRcds = allRetrievedResults;
        fillRelInfo();
    }

    public Evaluator(String qrelsFile, String resFile) throws Exception {
        this(qrelsFile, resFile, 1000);
    }

    public Evaluator(String qrelsFile, String resFile, int numTopDocs) throws Exception {
        relRcds = new AllRelRcds(qrelsFile);
        retRcds = new AllRetrievedResults(resFile);
        fillRelInfo(numTopDocs);
    }

    public Evaluator(String qrelsFile, Map<String, TopDocs> topDocsMap) {
        relRcds = new AllRelRcds(qrelsFile);
        retRcds = new AllRetrievedResults(topDocsMap);
        fillRelInfo();
    }

    public Evaluator(AllRelRcds relRcds, Map<String, TopDocs> topDocsMap) {
        this.relRcds = relRcds;
        retRcds = new AllRetrievedResults(topDocsMap);
        fillRelInfo();
    }

    public AllRetrievedResults getAllRetrievedResults() { return retRcds; }

    public AllRelRcds getRelRcds() { return relRcds; }

    public RetrievedResults getRetrievedResultsForQueryId(String qid) {
        return retRcds.getRetrievedResultsForQueryId(qid);
    }

    void fillRelInfo() {
        retRcds.fillRelInfo(relRcds);
    }

    void fillRelInfo(int numTopDocs) {
        retRcds.fillRelInfo(relRcds, numTopDocs);
    }

    public String computeAll() {
        return retRcds.computeAll();
    }

    public double compute(String qid, Metric m) {
        return retRcds.compute(qid, m);
    }

    @Override
    public String toString() {
        StringBuffer buff = new StringBuffer();
        buff.append(relRcds.toString()).append("\n");
        buff.append(retRcds.toString());
        return buff.toString();
    }

    public static void main(String[] args) {
        try {
            String qrelsFile = Constants.QRELS_TEST;
            String resFile = Constants.RES_FILE;

            Evaluator evaluator = new Evaluator(qrelsFile, resFile);
            System.out.println(evaluator.computeAll());
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }

    }

}
