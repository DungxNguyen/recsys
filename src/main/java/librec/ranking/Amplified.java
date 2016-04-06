package librec.ranking;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.util.Lists;
import librec.util.Logs;

public class Amplified extends SocialWithCF {

	public Amplified(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		// TODO Auto-generated constructor stub
	}

	public Map< Integer, Double > predict( int u, Set<Integer> jSet ){
		Map< Integer, Double > ratings = new HashMap< Integer, Double >();
		
		// find a number of similar users
		Map<Integer, Double> nns = new HashMap<>();
		
		SparseVector dv = userCorrs.row(u);
		for (int v : dv.getIndex()) {
			double sim = dv.get(v);

			nns.put(v, sim); 
		}

		// topN similar users
		if (knn > 0 && knn < nns.size()) {
			List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
			List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
			nns.clear();
			for (Map.Entry<Integer, Double> kv : subset)
				nns.put(kv.getKey(), kv.getValue()  );
		}
		
		// Add up social data
		SparseVector uv = socialMatrix.row( u );
		int numConns = uv.getCount();
		if( numConns == 0 ){
			// No connection
		}
		
		//-----------------------------------------------
//		Map<Integer, Double> nearestSocialNeighbor = new HashMap<>();
//		// For each neighbor in social network:
//		for (int v : uv.getIndex()) {
//			try{
//				nearestSocialNeighbor.put( v, uv.get( v ) );
//			}catch( IndexOutOfBoundsException e ){
////				e.printStackTrace();
//			}
//		}
//		
//		List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nearestSocialNeighbor, true);
//		int max = sorted.size() > knn ? knn : sorted.size();
//		int overlapped = 0;
//		for( int i = 0; i < max; i ++ ){
//			if( nns.containsKey( sorted.get( i ).getKey() ) ){
//				overlapped++;
////				nns.put( sorted.get( i ).getKey(), 2 * nns.remove( sorted.get( i ).getKey() ) );
//			}else{
//				try{
//					nns.put( sorted.get( i ).getKey(), dv.get( sorted.get( i ).getKey() ) );
//				}catch( IndexOutOfBoundsException e ){
//					Logs.debug( e.getMessage() );
//				}
//			}
//		}
//		if( max != 0 ){
//			overlappedRatio = ( overlappedRatio * testCount + (double) overlapped / max ) / (++testCount);
//		}
		//-----------------------------------------------
		
		//-----------------------------------------------
		// Amplified technique:
		double totalNumberOfInteractions = 0;
		for( int v : uv.getIndex() ){
			totalNumberOfInteractions += uv.get( v );
		}
		
		for( int v : uv.getIndex() ){
			if( nns.containsKey( v ) ){
				double newSimilarityValue = 0;
				newSimilarityValue = nns.get( v ) * ( 1 + uv.get(v) / totalNumberOfInteractions );
				if( newSimilarityValue > 1 ){
					newSimilarityValue = 1;
				}
				nns.remove( v );
				nns.put( v, newSimilarityValue );
			}
		}
		
		//-----------------------------------------------
		

		for( int j : jSet ){
			double rating = 0;
			for( Map.Entry<Integer, Double> kv : nns.entrySet() ){
				try{
					rating += kv.getValue() * trainMatrix.get( kv.getKey(), j );
				}catch( IndexOutOfBoundsException e ){
					// e.printStackTrace();
				}
			}
			ratings.put( j, rating );
		}
		
		return ratings;
	}
}
