package nnm.recsys;

import librec.main.LibRec;
import librec.util.Logs;

public class Network2Rec{

	public static final String[] CONFIG_FILES = new String[] { 

//			"UserKNNNetwork1Bi.conf",
//	 		"UserKNNNetwork1Uni.conf",
			"config/KNN/UserKNNNetwork2.conf", 
			"config/Social/UserKNNNetwork2_Social.conf" 			
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
