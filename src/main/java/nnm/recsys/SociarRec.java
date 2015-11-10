package nnm.recsys;

import librec.main.LibRec;
import librec.util.Logs;

public class SociarRec{

	public static final String[] CONFIG_FILES = new String[] { 
			"config/5/UserKNNNetwork2_Social.conf",
//			"config/5/UserKNNNetwork1Bi.conf",
			"config/10/UserKNNNetwork1Bi_Social.conf",
//			"config/10/UserKNNNetwork1Bi.conf",
			"config/15/UserKNNNetwork1Bi_Social.conf",
//			"config/15/UserKNNNetwork1Bi.conf",
			"config/20/UserKNNNetwork1Bi_Social.conf",
//			"config/20/UserKNNNetwork1Bi.conf",
//	 		"UserKNNNetwork1Uni_Social.conf",
//			"UserKNNNetwork2_Social.conf", 
//			"UserKNNNetwork3_Social.conf" 
			};
	// public static final String CONFIG_FILE = "UserKNN.conf";
//	public static final String[] CONFIG_FILES = new String[] { 
//			"UserKNNNetwork1Bi_Social.conf" };

	public static void main( String[] args ) throws Exception{
		// config logger
		Logs.config( "log4j_2.xml", true );

		LibRec librec = new LibRec();
		librec.setConfigFiles( CONFIG_FILES );
		// run algorithm
		librec.execute( args );
	}

}