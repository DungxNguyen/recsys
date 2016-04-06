// Copyright (C) 2014-2015 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.ranking;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.AbstractMap.SimpleImmutableEntry;

import librec.data.Configuration;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.DiagMatrix;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.VectorEntry;
import librec.intf.IterativeRecommender;
import librec.intf.Recommender.Measure;
import librec.util.FileIO;
import librec.util.Lists;
import librec.util.Logs;
import librec.util.Measures;
import librec.util.Stats;
import librec.util.Strings;

/**
 * <h3>WRMF: Weighted Regularized Matrix Factorization.</h3>
 * 
 * This implementation refers to the method proposed by Hu et al. at ICDM 2008.
 * 
 * <ul>
 * <li><strong>Binary ratings:</strong> Pan et al., One-class Collaborative Filtering, ICDM 2008.</li>
 * <li><strong>Real ratings:</strong> Hu et al., Collaborative filtering for implicit feedback datasets, ICDM 2008.</li>
 * </ul>
 * 
 * @author guoguibing
 * 
 */
@Configuration("binThold, alpha, factors, reg, regU, regI, numIters")
public class ALSWR extends IterativeRecommender {

	private float alpha;
	private double lambda;
	private boolean initMahout = true;

	public ALSWR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		isRankingPred = true; // item recommendation

		alpha = algoOptions.getFloat("-alpha");
		
		lambda = reg;
				
		// checkBinary();
	}
	
	@Override
	protected void initModel() throws Exception {

		P = new DenseMatrix(numUsers, numFactors);
		Q = new DenseMatrix(numItems, numFactors);

//		P.setAll( 1.0 );
//		Q.setAll( 1.0 );
		
		// initialize model
		if (initByNorm) {
			P.init(initMean, initStd);
			Q.init(initMean, initStd);
		} else if( initMahout ) {
			P.init();
			Q.init();
			DenseVector firstRow = new DenseVector( Q.numColumns() );
			firstRow.setAll( globalMean );
			Q.setRow( 0, firstRow );
		} else {
			P.init(); // P.init(smallValue);
			Q.init(); // Q.init(smallValue);
		}
		
		//TODO Initialize as Mahout

	}

	@Override
	protected void buildModel() throws Exception {

		// P is user matrix
		// Q is item matrix

		// Updating by using alternative least square (ALS)
		for (int iter = 1; iter <= numIters; iter++) {

			// Step 1: update user factors;
			for (int u = 0; u < numUsers; u++) {
//				for (int u = 0; u < 1; u++) {
				if (verbose && (u + 1) % 20 == 0)
					Logs.debug("{}{} runs at iteration = {}, user = {}/{}", algoName, foldInfo, iter, u + 1, numUsers );
//					Logs.debug("{}{} runs at iteration = {}, user = {}/{}, distance = {}", algoName, foldInfo, iter, u + 1, numUsers, P.mult(Q.transpose()).minus(trainMatrix).norm() );
				// Fix Q, compute P
				
				// Construct Q[u]
				
				// b_u
				SparseVector b_u = trainMatrix.row( u );
				
				// Items list of u:
				int[] uItems = b_u.getIndex();
				
				// Init Q[u]
				DenseMatrix Q_u = new DenseMatrix( numItems, numFactors );
				
				// Build Q[u]
				for( int i = 0; i < uItems.length; i++ ){
					Q_u.setRow( uItems[i], Q.row( uItems[i] ) );
				}
				
				// Calculate A_u
				DenseMatrix A_u = Q_u.transMult().add( Q.transMult() );
				
				// Calculate d_u
				DenseVector d_u = Q.add( Q_u ).transpose().mult( b_u );
				
				// Calculate p_u
				DenseMatrix W_u = A_u.add( DenseMatrix.eye(numFactors).scale( lambda  ) ).inv();
				
				DenseVector p_u = W_u.mult( d_u );
				
//				System.err.println( A_u );
//				
//				System.err.println( DenseMatrix.eye(numFactors).scale( lambda ) );
//				
//				System.err.println( W_u );
//				System.err.println( d_u );
//				
//				System.err.println( p_u );
				P.setRow( u, p_u );
			}

			// Step 2: update item factors;
			for (int i = 0; i < numItems; i++) {
				if (verbose && (i + 1) % 200 == 0)
					Logs.debug("{}{} runs at iteration = {}, item = {}/{}", algoName, foldInfo, iter, i + 1, numItems );
				// Fix P, compute Q
				
				// Construct P[i]
				SparseVector b_i = trainMatrix.column( i );
				
				// Users list of i:
				int[] iUsers = b_i.getIndex();
				
				// Init P[i]
				DenseMatrix P_i = new DenseMatrix( numUsers, numFactors );
				
				// Build P[i]
				for( int u = 0; u < iUsers.length; u++ ){
					P_i.setRow( iUsers[ u ], P.row( iUsers[ u ] ) );
				}
				
				// Calculate A_i
				DenseMatrix A_i = P_i.transMult().add( P.transMult() );
				
				// Calculate d_i
				DenseVector d_i = P.add( P_i ).transpose().mult( b_i );
				
				// Calculate Q_i
				DenseVector q_i = DenseMatrix.eye(numFactors).scale( lambda ).add( A_i ).inv().mult( d_i );
				
				Q.setRow( i, q_i );
			}
		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { binThold, alpha, numFactors, reg, numIters }, ",");
	}
	
	@Override
	/**
	 * @return the evaluation results of ranking predictions
	 */
	protected Map<Measure, Double> evalRankings() throws Exception {

		int capacity = Lists.initSize(testMatrix.numRows());

		List<Double> precs2 = new ArrayList<>(capacity);
		List<Double> recalls2 = new ArrayList<>(capacity);
		List<Double> precs5 = new ArrayList<>(capacity);
		List<Double> precs10 = new ArrayList<>(capacity);
		List<Double> recalls5 = new ArrayList<>(capacity);
		List<Double> recalls10 = new ArrayList<>(capacity);

		// candidate items for all users: here only training items
		// use HashSet instead of ArrayList to speedup removeAll() and contains() operations: HashSet: O(1); ArrayList: O(log n).
		Set<Integer> candItems = new HashSet<>(trainMatrix.columns());

		List<String> preds = null;
		String toFile = null;
		int numTopNRanks = numRecs < 0 ? 10 : numRecs;
		if (isResultsOut) {
			preds = new ArrayList<String>(1500);
			preds.add("# userId: recommendations in (itemId, ranking score) pairs, where a correct recommendation is denoted by symbol *."); // optional: file header
			toFile = tempDirPath
					+ String.format("%s-top-%d-items%s.txt", new Object[] { algoName, numTopNRanks, foldInfo }); // the output-file name
			FileIO.deleteFile(toFile); // delete possibly old files
		}

		if (verbose)
			Logs.debug("{}{} has candidate items: {}", algoName, foldInfo, candItems.size());

		// ignore items for all users: most popular items
		if (numIgnore > 0) {
			List<Map.Entry<Integer, Integer>> itemDegs = new ArrayList<>();
			for (Integer j : candItems) {
				itemDegs.add(new SimpleImmutableEntry<Integer, Integer>(j, trainMatrix.columnSize(j)));
			}
			Lists.sortList(itemDegs, true);
			int k = 0;
			for (Map.Entry<Integer, Integer> deg : itemDegs) {

				// ignore these items from candidate items
				candItems.remove(deg.getKey());
				if (++k >= numIgnore)
					break;
			}
		}

		// for each test user
		for (int u = 0, um = testMatrix.numRows(); u < um; u++) {

			if (verbose && ((u + 1) % 1000 == 0)){
				Logs.debug("{}{} evaluates progress: {} / {}", algoName, foldInfo, u + 1, um);
				Logs.debug("{},Current P2, P5, P10, R2, R5, R10: {},{},{},{},{},{}", foldInfo, Stats.mean(precs2), Stats.mean(precs5), Stats.mean(precs10),
										 Stats.mean(recalls2), Stats.mean(recalls5), Stats.mean(recalls10) );
			}

			// get positive items from test matrix
			List<Integer> testItems = testMatrix.getColumns(u);
			List<Integer> correctItems = new ArrayList<>();

			// intersect with the candidate items
			for (Integer j : testItems) {
				if (candItems.contains(j))
					correctItems.add(j);
			}

			if (correctItems.size() == 0)
				continue; // no testing data for user u

			// remove rated items from candidate items
			List<Integer> ratedItems = trainMatrix.getColumns(u);

			// predict the ranking scores (unordered) of all candidate items
			List<Map.Entry<Integer, Double>> itemScores = new ArrayList<>(Lists.initSize(candItems));
			for (final Integer j : candItems) {
				// item j is not rated 
				if (!ratedItems.contains(j)) {
					final double rank = ranking(u, j);
//					final double rank = 1.0;
					if (!Double.isNaN(rank)) {
						itemScores.add(new SimpleImmutableEntry<Integer, Double>(j, rank));
					}
				} 
			}

//			for( Map.Entry< Integer, Double > kv : predict( u, candItems ).entrySet() ){
//				if( !ratedItems.contains( kv.getKey() ) && !Double.isNaN( kv.getValue() ) ){
//					itemScores.add( kv );
//				}
//			}
			
			if (itemScores.size() == 0)
				continue; // no recommendations available for user u

			// order the ranking scores from highest to lowest: List to preserve orders
			Lists.sortList(itemScores, true);
			List<Map.Entry<Integer, Double>> recomd = (numRecs <= 0 || itemScores.size() <= numRecs) ? itemScores
					: itemScores.subList(0, numRecs);

			List<Integer> rankedItems = new ArrayList<>();
			StringBuilder sb = new StringBuilder();
			int count = 0;
			for (Map.Entry<Integer, Double> kv : recomd) {
				Integer item = kv.getKey();
				rankedItems.add(item);

				if (isResultsOut && count < numTopNRanks) {
					// restore back to the original item id
					sb.append("(").append(rateDao.getItemId(item));

					if (testItems.contains(item))
						sb.append("*"); // indicating correct recommendation

					sb.append(", ").append(kv.getValue().floatValue()).append(")");

					count++;

					if (count < numTopNRanks)
						sb.append(", ");
				}
			}

			List<Integer> cutoffs = Arrays.asList(2, 5, 10);
			Map<Integer, Double> precs = Measures.PrecAt(rankedItems, correctItems, cutoffs);
			Map<Integer, Double> recalls = Measures.RecallAt(rankedItems, correctItems, cutoffs);

			precs2.add(precs.get(2));
			recalls2.add(recalls.get(2));
			precs5.add(precs.get(5));
			precs10.add(precs.get(10));
			recalls5.add(recalls.get(5));
			recalls10.add(recalls.get(10));


			// output predictions
			if (isResultsOut) {
				// restore back to the original user id
				preds.add(rateDao.getUserId(u) + ": " + sb.toString());
				if (preds.size() >= 1000) {
					FileIO.writeList(toFile, preds, true);
					preds.clear();
				}
			}
		}
		// write results out first
		if (isResultsOut && preds.size() > 0) {
			FileIO.writeList(toFile, preds, true);
			Logs.debug("{}{} has writeen item recommendations to {}", algoName, foldInfo, toFile);
		}

		// measure the performance
		Map<Measure, Double> measures = new HashMap<>();
		measures.put(Measure.Pre2, Stats.mean(precs2));
		measures.put(Measure.Rec2, Stats.mean(recalls2));
		measures.put(Measure.Pre5, Stats.mean(precs5));
		measures.put(Measure.Pre10, Stats.mean(precs10));
		measures.put(Measure.Rec5, Stats.mean(recalls5));
		measures.put(Measure.Rec10, Stats.mean(recalls10));

		return measures;
	}
}
