// Copyright (C) 2014 Guibing Guo
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

package librec.rating;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.Map.Entry;

import librec.data.Configuration;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.SymmMatrix;
import librec.intf.Recommender;
import librec.intf.Recommender.Measure;
import librec.util.FileIO;
import librec.util.Lists;
import librec.util.Logs;
import librec.util.Measures;
import librec.util.Stats;
import librec.util.Strings;

/**
 * <h3>User-based Nearest Neighbors</h3>
 * 
 * <p>
 * It supports both recommendation tasks: (1) rating prediction; and (2) item ranking (by configuring
 * {@code item.ranking=on} in the librec.conf). For item ranking, the returned score is the summation of the
 * similarities of nearest neighbors.
 * </p>
 * 
 * <p>
 * When the number of users is extremely large which makes it memory intensive to store/precompute all user-user
 * correlations, a trick presented by (Jahrer and Toscher, Collaborative Filtering Ensemble, JMLR 2012) can be applied.
 * Specifically, we can use a basic SVD model to obtain user-feature vectors, and then user-user correlations can be
 * computed by Eqs (17, 15).
 * </p>
 * 
 * @author guoguibing
 * 
 */
@Configuration("knn, similarity, shrinkage")
public class UserKNN2 extends Recommender {

	// user: nearest neighborhood
	private SymmMatrix userCorrs;
	private DenseVector userMeans;

	public UserKNN2(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	@Override
	protected void initModel() throws Exception {
		userCorrs = buildCorrs(true);
		userMeans = new DenseVector(numUsers);
		for (int u = 0; u < numUsers; u++) {
			SparseVector uv = trainMatrix.row(u);
			userMeans.set(u, uv.getCount() > 0 ? uv.mean() : globalMean);
		}
	}

	@Override
	protected double predict(int u, int j) {

		// find a number of similar users
		Map<Integer, Double> nns = new HashMap<>();

		SparseVector dv = userCorrs.row(u);
		for (int v : dv.getIndex()) {
			double sim = dv.get(v);
			double rate = trainMatrix.get(v, j);

			if (isRankingPred && rate > 0)
				nns.put(v, sim); // similarity could be negative for item ranking
			else if (sim > 0 && rate > 0)
				nns.put(v, sim);
		}

		// topN similar users
		if (knn > 0 && knn < nns.size()) {
			List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
			List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
			nns.clear();
			for (Map.Entry<Integer, Double> kv : subset)
				nns.put(kv.getKey(), kv.getValue() );
		}
		
		//Alternative
		//-------------------------------------------------------------
//		for (int v : dv.getIndex()) {
//			double sim = dv.get(v);
//			double rate = trainMatrix.get(v, j);
//
//			if (isRankingPred && rate >= 0)
//				nns.put(v, sim); // similarity could be negative for item ranking
//			else if (sim > 0 && rate >= 0)
//				nns.put(v, sim);
//		}
//
//		// topN similar users
//		if (knn > 0 && knn < nns.size()) {
//			List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
//			List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
//			nns.clear();
//			for (Map.Entry<Integer, Double> kv : subset)
//				nns.put(kv.getKey(), kv.getValue() * trainMatrix.get( kv.getKey(), j ) );
//		}
		//-------------------------------------------------------------

		if (nns.size() == 0)
			return isRankingPred ? 0 : globalMean;

		if (isRankingPred) {
			// for item ranking

			return Stats.sum(nns.values());
		} else {
			// for rating prediction

			double sum = 0, ws = 0;
			for (Entry<Integer, Double> en : nns.entrySet()) {
				int v = en.getKey();
				double sim = en.getValue();
				double rate = trainMatrix.get(v, j);

				sum += sim * (rate - userMeans.get(v));
				ws += Math.abs(sim);
			}

			return ws > 0 ? userMeans.get(u) + sum / ws : globalMean;
		}
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
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

			if (verbose && ((u + 1) % 1 == 0))
				Logs.debug("{}{} evaluates progress: {} / {}", algoName, foldInfo, u + 1, um);
				Logs.debug("{},Current P2, P5, P10, R2, R5, R10: {},{},{},{},{},{}", foldInfo, Stats.mean(precs2), Stats.mean(precs5), Stats.mean(precs10),
										 Stats.mean(recalls2), Stats.mean(recalls5), Stats.mean(recalls10) );

			// number of candidate items for all users
			int numCands = candItems.size();

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
				} else {
					numCands--;
				}
			}

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

			int numDropped = numCands - rankedItems.size();

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
