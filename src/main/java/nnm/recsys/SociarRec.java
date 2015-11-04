package nnm.recsys;

import librec.main.LibRec;
import librec.util.Logs;

public class SociarRec{

	public static final String[] CONFIG_FILES = new String[] { 
			"UserKNNNetwork1Bi_Social.conf",
	 		"UserKNNNetwork1Uni_Social.conf",
			"UserKNNNetwork2_Social.conf", 
			"UserKNNNetwork3_Social.conf" };
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