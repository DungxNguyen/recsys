package nnm.recsys.jointnetwork;

import java.util.AbstractMap.SimpleImmutableEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import librec.data.DataDAO;
import librec.data.SparseMatrix;
import librec.intf.Recommender;
import librec.ranking.Amplified;
import librec.ranking.Social;
import librec.util.FileConfiger;
import librec.util.FileIO;
import librec.util.LineConfiger;
import librec.util.Lists;
import librec.util.Logs;
import librec.util.Measures;
import librec.util.Stats;

public class JointNetwork extends Social {
	
	private double[] implicitWeights = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	
	private Amplified explicitRecommender;
	private String explicitConfigFile;
	
	public JointNetwork(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		
		setAlgoName( "JOINT" );
		explicitConfigFile = algoOptions.getString("-explicit");
		
		FileConfiger cf = Recommender.cf;
		DataDAO rateDao = Recommender.rateDao;
		try {
			buildExplicitRecommender();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		Recommender.cf = cf;
		Recommender.rateDao = rateDao;
	}
	
	public void buildExplicitRecommender() throws Exception{
		FileConfiger cf = new FileConfiger( explicitConfigFile );

		// seeding the general recommender
		Recommender.cf = cf;
		
		// DAO object
		DataDAO rateDao = new DataDAO(cf.getPath("dataset.ratings"), Recommender.rateDao.getUserIds(), Recommender.rateDao.getItemIds());

		// data configuration
		LineConfiger ratingOptions = cf.getParamOptions("ratings.setup");

		// data columns to use
		List<String> cols = ratingOptions.getOptions("-columns");
		int[] columns = new int[cols.size()];
		for (int i = 0; i < cols.size(); i++)
			columns[i] = Integer.parseInt(cols.get(i));

		// is first line: headline
		rateDao.setHeadline(ratingOptions.contains("-headline"));

		// rating threshold
		float binThold = ratingOptions.getFloat("-threshold");

		// time unit of ratings' timestamps
		TimeUnit timeUnit = TimeUnit.valueOf(ratingOptions.getString("--time-unit", "seconds").toUpperCase());
		rateDao.setTimeUnit(timeUnit);

		SparseMatrix[] data = ratingOptions.contains("--as-tensor") ? rateDao.readTensor(columns, binThold) : rateDao
				.readData(columns, binThold);
		SparseMatrix rateMatrix = data[0];
		
//		Recommender.rateDao = rateDao;
		explicitRecommender = new Amplified(rateMatrix, null, 0);
		explicitRecommender.initModel();
	}
	
	public Map< Integer, Double > predict( int u, Set<Integer> jSet ){		
		Map< Integer, Double > implicitRatings = super.predict(u, jSet);
		Map< Integer, Double > explicitRatings = explicitRecommender.predict( u, jSet );
		

		return joint( u, jSet, implicitRatings, explicitRatings, 0.5 );
	}
	
	public Map< Integer, Double > joint( int u, Set< Integer > jSet, Map<Integer, Double> implicitRatings, Map<Integer, Double> explicitRatings, double implicitWeight ){
		Map< Integer, Double > ratings = new HashMap< Integer, Double >();
		if( socialMatrix.row( u ).size() == 0 ){
			ratings = explicitRatings;
		}else if( explicitRecommender.socialMatrix.row( u ).size() == 0 ){
			ratings = implicitRatings;
		}else{
			for( int j : jSet ){
				int contextCount = 0;
				double implicitScore = 0.0;
				double explicitScore = 0.0;
				if( implicitRatings.containsKey( j ) ){
					contextCount++;
					implicitScore = implicitRatings.get( j );
				}
				if( explicitRatings.containsKey( j ) ){
					contextCount++;
					explicitScore = explicitRatings.get( j );
				}
				ratings.put( j, implicitScore * implicitWeight * contextCount +
								   explicitScore * (1 - implicitWeight) * contextCount );
			}
		}
		return ratings;
	}
	
	@Override
	/**
	 * @return the evaluation results of ranking predictions
	 */
	protected Map<Measure, Double> evalRankings() throws Exception {
		List<ArrayList<Double>> precs10 = new ArrayList<ArrayList<Double>>();
		List<ArrayList<Double>> recalls10 = new ArrayList<ArrayList<Double>>();
		for( int i = 1; i <= 9; i++ ){
			precs10.add( new ArrayList<Double>() );
			recalls10.add( new ArrayList< Double >() );
		}

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
//				Logs.debug("{} Current P2, P5, P10, R2, R5, R10: {},{},{},{},{},{}", foldInfo, Stats.mean(precs10), Stats.mean(recalls10) );
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
			
			// Final ratings
			Map< Integer, Double > implicitRatings = super.predict(u, candItems);
			Map< Integer, Double > explicitRatings = explicitRecommender.predict( u, candItems );
			
			for( int i = 1; i <=9; i++ ){
			
				// predict the ranking scores (unordered) of all candidate items
				List<Map.Entry<Integer, Double>> itemScores = new ArrayList<>(Lists.initSize(candItems));		
				for( Map.Entry< Integer, Double > kv : joint( u, candItems, implicitRatings, explicitRatings, implicitWeights[i-1] ).entrySet() ){
					if( !ratedItems.contains( kv.getKey() ) && !Double.isNaN( kv.getValue() ) ){
						itemScores.add( kv );
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
	
				List<Integer> cutoffs = Arrays.asList(2, 5, 10);
				Map<Integer, Double> precs = Measures.PrecAt(rankedItems, correctItems, cutoffs);
				Map<Integer, Double> recalls = Measures.RecallAt(rankedItems, correctItems, cutoffs);
	
				precs10.get(i - 1).add(precs.get(10));
				recalls10.get(i - 1).add(recalls.get(10));
	
	
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
		}

		// write results out first
		if (isResultsOut && preds.size() > 0) {
			FileIO.writeList(toFile, preds, true);
			Logs.debug("{}{} has writeen item recommendations to {}", algoName, foldInfo, toFile);
		}

		// measure the performance
		Map<Measure, Double> measures = new HashMap<>();
		for( int i = 1; i <= 9; i++ ){
			double precision = Stats.mean( precs10.get(i - 1 ) ) ;
			double recall = Stats.mean( recalls10.get( i - 1 ) );
			Measure m;
			switch ( i ){
			case 1: m = Measure.F1_1;break;
			case 2: m = Measure.F1_2;break;
			case 3: m = Measure.F1_3;break;
			case 4: m = Measure.F1_4;break;
			case 5: m = Measure.F1_5;break;
			case 6: m = Measure.F1_6;break;
			case 7: m = Measure.F1_7;break;
			case 8: m = Measure.F1_8;break;
			case 9: m = Measure.F1_9;break;
			default: m = Measure.F1;
			}
			measures.put( m, 2 * precision * recall / ( precision + recall ));
		}

		return measures;
	}
	
	@Override
	/**
	 * @return the evaluation information of a recommend
	 */
	public String getEvalInfo(Map<Measure, Double> measures)  {
		String evalInfo = null;
		if (isRankingPred) {
			if (isDiverseUsed)
				evalInfo = String.format("%6f,%6f,%6f,%6f,%6f,%6f,%6f,%6f,%6f", 
						measures.get(Measure.F1_1),measures.get(Measure.F1_2),measures.get(Measure.F1_3)
						,measures.get(Measure.F1_4),measures.get(Measure.F1_5),measures.get(Measure.F1_6)
						,measures.get(Measure.F1_7),measures.get(Measure.F1_8),measures.get(Measure.F1_9));
			else
				evalInfo = String.format("%6f,%6f,%6f,%6f,%6f,%6f,%6f,%6f,%6f", 
						measures.get(Measure.F1_1),measures.get(Measure.F1_2),measures.get(Measure.F1_3)
						,measures.get(Measure.F1_4),measures.get(Measure.F1_5),measures.get(Measure.F1_6)
						,measures.get(Measure.F1_7),measures.get(Measure.F1_8),measures.get(Measure.F1_9));

		} else {
			evalInfo = String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", measures.get(Measure.MAE),
					measures.get(Measure.RMSE), measures.get(Measure.NMAE), measures.get(Measure.rMAE),
					measures.get(Measure.rRMSE), measures.get(Measure.MPE));

			// for some graphic models
			if (measures.containsKey(Measure.Perplexity)) {
				evalInfo += String.format(",%.6f", measures.get(Measure.Perplexity));
			}
		}

		return evalInfo;
	}
}
