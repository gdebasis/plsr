import org.apache.lucene.index.*;
import org.apache.lucene.util.BytesRef;

import java.io.*;
import java.util.*;

public class RelEstimator {
    IndexReader reader;
    Map<String, Integer> Rw = new HashMap<>();
    String betaFile;
    AllRelRcds allRels;

    private long totalRelevantDocs = 0;
    private final long totalDocs;

    public RelEstimator(IndexReader reader, String qrelsFile, String betaFile) {
        this.reader = reader;
        this.totalDocs = reader.numDocs();
        allRels = new AllRelRcds(qrelsFile);
        this.betaFile = betaFile;

        try {
            computeRw();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    /**
     * Compute R_w using AllRelRcds and your docID ↔ offset functions
     */
    public void computeRw() throws IOException {

        if (modelWeightsExist()) {
            System.out.println("Loading precomputed r(w) weights from " + betaFile);
            Rw = loadRw(betaFile);
        }
        else {
            System.out.println("Computing r(w) rel priors...");

            for (PerQueryRelDocs perQuery : allRels.perQueryRels.values()) {
                for (String docId : perQuery.getRelDocs()) {
                    if (!perQuery.isRel(docId, 1)) continue;

                    int luceneDocId = IndexUtils.getDocOffsetFromId(docId);
                    if (luceneDocId < 0) continue;

                    Terms terms = reader.getTermVector(luceneDocId, Constants.CONTENT_FIELD);
                    if (terms == null) continue;

                    totalRelevantDocs++;

                    TermsEnum termsEnum = terms.iterator();
                    BytesRef term;

                    while ((term = termsEnum.next()) != null) {
                        String w = term.utf8ToString();
                        Rw.merge(w, 1, Integer::sum);
                    }
                }
            }
            saveRw(Rw, this.betaFile);
        }
    }

    /**
     * Document frequency
     */
    public int getNw(String term) throws IOException {
        return reader.docFreq(new Term(Constants.CONTENT_FIELD, term));
    }

    /**
     * RSJ beta with smoothing
     */
    public double getBeta(String term) throws IOException {
        int rw = Rw.getOrDefault(term, 0);
        int nw = getNw(term);

        double p_w_R = (rw + 0.5) / (totalRelevantDocs + 1.0);
        double p_w_notR = ((nw - rw) + 0.5) /
                ((totalDocs - totalRelevantDocs) + 1.0);

        return Math.log(1 + p_w_R / p_w_notR);
        //return p_w_R / p_w_notR;
    }

    /**
     * Compute betas for observed terms
     */
    public Map<String, Double> computeBetas() throws IOException {
        Map<String, Double> beta = new HashMap<>();

        for (String term : Rw.keySet()) {
            beta.put(term, getBeta(term));
        }
        return beta;
    }

    public boolean modelWeightsExist() {
        File f = new File(betaFile);
        return (f.exists() && f.length() > 0);
    }

    public static void saveRw(Map<String, Integer> tfMap,
                                String filePath) throws IOException {

        BufferedWriter bw = new BufferedWriter(new FileWriter(filePath));

        for (Map.Entry<String, Integer> e : tfMap.entrySet()) {
            bw.write(e.getKey());
            bw.write("\t");
            bw.write(Integer.toString(e.getValue()));
            bw.newLine();
        }

        bw.close();
    }

    public static Map<String, Integer> loadRw(String filePath) throws IOException {
        Map<String, Integer> tfMap = new HashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(filePath));
        String line;

        while ((line = br.readLine()) != null) {
            String[] parts = line.split("\t");
            if (parts.length != 2) continue;

            String term = parts[0];
            int value = Integer.parseInt(parts[1]);

            tfMap.put(term, value);
        }

        br.close();
        return tfMap;
    }
}