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

import java.util.Arrays;

import librec.data.Configuration;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.DiagMatrix;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.data.VectorEntry;
import librec.intf.IterativeRecommender;
import librec.util.Logs;
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
				if (verbose && (u + 1) % 200 == 0)
					Logs.debug("{}{} runs at iteration = {}, user = {}/{}, distance = {}", algoName, foldInfo, iter, u + 1, numUsers, P.mult(Q.transpose()).minus(trainMatrix).norm() );
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
					Logs.debug("{}{} runs at iteration = {}, item = {}/{}, distance = {}", algoName, foldInfo, iter, i + 1, numUsers, P.mult(Q.transpose()).minus(trainMatrix).norm() );
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

}
