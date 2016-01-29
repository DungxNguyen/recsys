package nnm.recsys.contentbase;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.AbstractMap.SimpleImmutableEntry;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.SocialRecommender;
import librec.intf.Recommender.Measure;
import librec.util.FileIO;
import librec.util.Lists;
import librec.util.Logs;
import librec.util.Measures;
import librec.util.Stats;

public class ContentBaseRecommender extends SocialRecommender {

	private static final String USER_TAG_DATA_FILE = "user_tag_processed.csv";
	private static final String TITLE_ABSTRACT = "titleabstractprocessedallpapers.csv";

	private HashMap<String, Integer> dictionary;
	private SparseMatrix user_matrix;
	private SparseMatrix paper_matrix;
	private HashMap<Integer, HashSet<Integer>> userCandidateList;

	private HashMap<Integer, HashMap<Integer, HashSet< String>>> userTagData;
	private HashMap<Integer, String> titleAbstract;

	public ContentBaseRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	@Override
	protected void initModel() throws Exception {
		userTagData = readUserTagData(USER_TAG_DATA_FILE);
		titleAbstract = readTitleAbstract(TITLE_ABSTRACT);
		
		dictionary = constructDictionary( titleAbstract );

		paper_matrix = constructPaperMatrix(titleAbstract, dictionary );
		userCandidateList = constructUserCandidateList( userTagData );
		user_matrix = constructUserMatrix(userCandidateList, paper_matrix);
	}


	protected HashMap<Integer, HashMap<Integer, HashSet< String>>> readUserTagData(String userTagDataFile) {
		HashMap<Integer, HashMap<Integer, HashSet< String >>> userTagData = new HashMap<Integer, HashMap<Integer, HashSet< String>>>();

		try {
			BufferedReader reader = new BufferedReader(new FileReader(new File(userTagDataFile)));

			String line;
			while( (line = reader.readLine()) != null ){
				String[] elements = line.split("[\",]+");
				if( !rateDao.isExistedRawUserId( elements[1] ) || !rateDao.isExistedRawItemId( elements[2] ) ){
					continue;
				}
				int userId = rateDao.getUserId( elements[1] );
				int paperId = rateDao.getItemId( elements[2] );
				if ( !userTagData.containsKey(userId) ) {
					userTagData.put(userId, new HashMap<Integer, HashSet< String>>());
				}
				if( !userTagData.get( userId ).containsKey( paperId ) ){
					userTagData.get( userId ).put( paperId, new HashSet< String >() );
				}
				userTagData.get( userId ).get( paperId ).add( elements[3] );
			}

			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return userTagData;
	}

	protected HashMap<Integer, String> readTitleAbstract(String titleAbstractFile) {
		HashMap<Integer, String> titleAbstract = new HashMap<Integer, String>();

		try {
			BufferedReader reader = new BufferedReader( new FileReader( new File( titleAbstractFile ) ) );
			String line;
			while ( (line = reader.readLine() ) != null ) {
				String[] elements = line.split("[\",]+");
				if( rateDao.isExistedRawItemId( elements[1] ) ){
					int paperId = rateDao.getItemId( elements[1] );
					titleAbstract.put( paperId, elements[2] );
				}
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return titleAbstract;
	}
	
	protected HashMap< String, Integer > constructDictionary( HashMap< Integer, String > titleAbstract ){
		HashMap<String, Integer > dictionary = new HashMap< String, Integer >();
		
		int index = 0;
		for( Map.Entry<Integer, String> entry : titleAbstract.entrySet() ){
			String elements[] = entry.getValue().split( "\\s+" );
			for( String element : elements ){
				if( !dictionary.containsKey( element ) ){
					dictionary.put( element, index++ );
				}
			}
		}
		
		return dictionary;
	}
	
	protected SparseMatrix constructPaperMatrix( HashMap< Integer, String > titleAbstract, HashMap< String, Integer> dictionary ){
		Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
		Multimap<Integer, Integer> colMap = HashMultimap.create();
		
		for( Map.Entry< Integer, String > entry : titleAbstract.entrySet() ){
			int paperId = entry.getKey();
			for( String element : entry.getValue().split( "\\s+") ){
				dataTable.put( paperId, dictionary.get( element ), 1.0 );
				colMap.put( dictionary.get( element ), paperId );
			}
		}
		
		SparseMatrix paperMatrix = new SparseMatrix( numItems, dictionary.size(), dataTable, colMap );
		return paperMatrix;
	}

	protected HashMap<Integer, HashSet<Integer>> constructUserCandidateList( HashMap<Integer, HashMap<Integer, HashSet< String>>> userTagData ) {
		HashMap< Integer, HashSet< Integer > > candidates = new HashMap< Integer, HashSet<Integer> >();
		
		for( int u = 0; u < numUsers; u++ ){
			HashSet< Integer > userPaper = new HashSet< Integer >();
			HashSet< String > userTag = new HashSet< String >();
			for( int index : trainMatrix.row( u ).getIndex() ){
				userPaper.add( index );
			}
			if( userTagData.containsKey( u ) ){
				for( Map.Entry< Integer, HashSet< String >> entry : userTagData.get( u ).entrySet() ){
					userTag.addAll( entry.getValue() );
				}
			}
			
			SparseVector uNeighbor = socialMatrix.row(u);
			for( int v : uNeighbor.getIndex() ){
				int[] vPaper = trainMatrix.row( v ).getIndex();
				for( int paper : vPaper ){
					if( !userPaper.contains( paper ) ){
						if( userTagData.containsKey(v) && userTagData.get( v ).containsKey( paper ) ){
							for( String tag : userTagData.get( v ).get( paper ) ){
								if( userTag.contains( tag ) ){
									if( candidates.containsKey( u ) ){
										candidates.get( u ).add( paper );
									}else{
										candidates.put( u, new HashSet< Integer >() );
										candidates.get( u ).add( paper );
									}
								}
							}
						}
					}
				}
			}
		}
		
		return candidates;
	}
	
	protected SparseMatrix constructUserMatrix( HashMap<Integer, HashSet<Integer>> userCandidateList, SparseMatrix paper_matrix ){
		Table<Integer, Integer, Double> dataTable = HashBasedTable.create();
		Multimap<Integer, Integer> colMap = HashMultimap.create();
		
		for( int u = 0; u < numUsers; u++ ){
			if( !userCandidateList.containsKey( u ) ){
				continue;
			}
			SparseVector userVector = new SparseVector( paper_matrix.numColumns() );
			for( int paper : userCandidateList.get( u ) ){
				for( int index : paper_matrix.row( paper ).getIndex() ){
					userVector.set( index, 1.0 );
				}
			}
			for( int index : userVector.getIndex() ){
				dataTable.put( u, index, 1.0 );
				colMap.put( index, u );
			}
		}
		
		SparseMatrix userMatrix = new SparseMatrix( numUsers, paper_matrix.numColumns(), dataTable, colMap);
		return userMatrix;
	}
	
	private double cosine( SparseVector iv, SparseVector jv ){
		return iv.inner(jv) / (Math.sqrt(iv.inner(iv)) * Math.sqrt(jv.inner(jv)));
	}
	
	protected Map<Integer, Double> predict( int u, Set<Integer> jSet ){
		Map<Integer, Double> ratings = new HashMap< Integer, Double>();
		
		for( int index : jSet ){
			ratings.put( index, cosine( user_matrix.row(u), paper_matrix.row(index) ) );
		}
		
		return ratings;
	}
	
	@Override
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

			if( verbose ){
//			if (verbose && ((u + 1) % 1000 == 0)){
				Logs.debug("{}{} evaluates progress: {} / {}", algoName, foldInfo, u + 1, um);
				Logs.debug("{} Current P2, P5, P10, R2, R5, R10: {},{},{},{},{},{}", foldInfo, Stats.mean(precs2), Stats.mean(precs5), Stats.mean(precs10),
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
//			for (final Integer j : candItems) {
//				// item j is not rated 
//				if (!ratedItems.contains(j)) {
//					final double rank = ranking(u, j);
////					final double rank = 1.0;
//					if (!Double.isNaN(rank)) {
//						itemScores.add(new SimpleImmutableEntry<Integer, Double>(j, rank));
//					}
//				} else {
//					numCands--;
//				}
//			}
//			
			for( Map.Entry< Integer, Double > kv : predict( u, candItems ).entrySet() ){
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
