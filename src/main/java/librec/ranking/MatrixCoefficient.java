package librec.ranking;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Set;

import librec.data.Configuration;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.SymmMatrix;
import librec.intf.SocialRecommender;

@Configuration("knn, similarity, shrinkage")
public class MatrixCoefficient extends SocialRecommender{

	// user: nearest neighborhood
	private SymmMatrix jaccardCorrs;
	private SymmMatrix logllhCorrs;

	
	static{
		resetStatics = false;
	}
	
	public MatrixCoefficient(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	@Override
	protected void initModel() throws Exception {
		similarityMeasure = "jaccard-binary";
		jaccardCorrs = buildCorrs( true );

		similarityMeasure = "logllh-binary";
		logllhCorrs = buildCorrs( true );
	}
	
	@Override
	public void printInfo( String fileName ){
		try{
			PrintWriter printer = new PrintWriter( new BufferedWriter( new FileWriter( new File( fileName ) ) ) );
			
			for (int u = 0; u < numUsers; u++) {
				String invertedU = rateDao.getUserId( u );
				SparseVector uvJaccard = jaccardCorrs.row(u);
				SparseVector uvLogllh = logllhCorrs.row(u);
				
//				Set<Integer> vIndex = new HashSet<Integer>();
//				vIndex.addAll(uvJaccard.getIndexList());
//				vIndex.addAll(uvLogllh.getIndexList());
				
				SparseVector uSocialRelations = socialMatrix.row( u );
				
				for( int v : uSocialRelations.getIndex() ){
					String invertedV = rateDao.getUserId( v );
//					if( vIndex.contains( v ) ){
						printer.println( invertedU + "," + invertedV + "," + uvLogllh.get(v) + "," + uvJaccard.get(v) );
//						printer.println( invertedU + "," + invertedV + "," + "," + uvJaccard.get(v) );
//					}else{
//						printer.println( invertedU + "," + invertedV + "," + 0 + "," + 0 );
//					}
				}
			}
			
			printer.close();
		}catch( Exception e ){
			e.printStackTrace();
		}
	}
}
