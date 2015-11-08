package nnm.recsys;

import librec.main.LibRec;
import librec.util.Logs;

public class SimilarityRec{

	public static final String[] CONFIG_FILES = new String[] { 
			"config/Similarity/UserKNNNetwork1Bi.conf",
	 		"config/Similarity/UserKNNNetwork1Bi_COS.conf",
	 		"config/Similarity/UserKNNNetwork1Bi_COSBI.conf",
			"config/Similarity/UserKNNNetwork1Bi_CPC.conf",
			"config/Similarity/UserKNNNetwork1Bi_MSD.conf",
			"config/Similarity/UserKNNNetwork1Bi_PCC.conf",
			"config/Similarity/UserKNNNetwork1Bi_LLLH.conf",
			"config/Similarity/UserKNNNetwork1Bi_JACBIN.conf"

			};
	
//	public static final String[] CONFIG_FILES = new String[] { 
//			"UserKNNNetwork1Bi.conf" };
	// public static final String CONFIG_FILE = "UserKNN.conf";

	public static void main( String[] args ) throws Exception{
		// config logger
		Logs.config( "log4j.xml", true );

		LibRec librec = new LibRec();
		librec.setConfigFiles( CONFIG_FILES );
		// run algorithm
		librec.execute( args );
	}

}
