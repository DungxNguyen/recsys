package nnm.recsys;

import java.io.File;

import librec.main.LibRec;
import librec.util.Logs;

public class NRec{

//	public static final String[] CONFIG_FILES = new String[] { 
//
////			"UserKNNNetwork1Bi.conf",
////	 		"UserKNNNetwork1Uni.conf",
//			"UserKNNNetwork2.conf", 
//			"UserKNNNetwork3.conf" 			
//			};
	
//	public static final String[] CONFIG_FILES = new String[] { 
//			"UserKNNNetwork1Bi.conf" };
	// public static final String CONFIG_FILE = "UserKNN.conf";

	public static void main( String[] args ) throws Exception{
		// config logger
		Logs.config( "log4j.xml", true );
		
		String configDirectoryName = args[0];
		System.out.println( configDirectoryName );
		
		File configDirectory = new File( configDirectoryName );
		String[] configFiles = new String[ 100 ];
		
		int i = 0;
		for( String file : configDirectory.list() ){
			configFiles[i++] = configDirectoryName + "/" + file;
			System.out.println(  configFiles[i-1] );
		}

		LibRec librec = new LibRec();
		librec.setConfigFiles( configFiles );
		// run algorithm
		librec.execute( null );
	}

}
