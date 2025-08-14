
/**
 * @file
 * @author andy@impv.au
 * @version 1.0
 * @brief Entry point and harness for the memetic algorithm
 * 
 * @bug Need to add .devcontainer and .vscode configurations files to the list of files 
 * @bug Need to change all clone() functions to copy constructors
 * 
 */

// Local
#include <memetico/args.h>
#include <memetico/globals.h>
#include <memetico/helpers/rng.h>
#include <memetico/data/data_set.h>
#include <memetico/optimise/objective.h>
#include <memetico/models/regression.h>
#include <memetico/models/cont_frac_dd.h>
#include <memetico/models/branch_cont_frac_dd.h>
#include <memetico/population/pop.h>
#include <memetico/global_types.h>
#include <mpi.h>

// Std
#include <cstdlib>

using namespace std;

// Logic Globals
bool            meme::GPU = false;
uint_fast32_t   meme::SEED = 42;
size_t          meme::GENERATIONS = 200;
double          meme::MUTATE_RATE = 0.2;
size_t          meme::LOCAL_SEARCH_INTERVAL = 1;
size_t          meme::STALE_RESET = 5;
bool            meme::INT_ONLY = false;
size_t          meme::LOCAL_SEARCH_RUNS = 4;
size_t          meme::NELDER_MEAD_STALE = 10;
size_t          meme::NELDER_MEAD_MOVES = 1500;
double          meme::LOCAL_SEARCH_DATA_PCT = 0;
double          meme::PENALTY = 0;
size_t          meme::GEN = 0;
long int        meme::MAX_TIME = 10*60;
long int        meme::RUN_TIME = 0;
double          meme::EPSILON = 0;

size_t          meme::DEPTH = 4;
size_t          meme::POCKET_DEPTH = 1;
size_t          meme::DIVERSITY_COUNT = 3;

// Derivative Globals
size_t          meme::IFR = 0;
string          meme::IN_DER = "exact";
size_t          meme::MAX_DER_ORD = 3;
// Least Squares Globals
// To-do: Implement a way to set this in args
size_t          meme::NUM_NEIGHBORS = 16;

DynamicDepthType meme::DYNAMIC_DEPTH_TYPE = DynamicNone;

// File Globals
string          meme::TRAIN_FILE = "sinx.csv";
string          meme::TEST_FILE = "sinx.csv";
string          meme::LOG_DIR = "out/";
ofstream        meme::master_log;

// Technical Globals
size_t          meme::PREC = 18;
bool            meme::DEBUG = false;

// Global Heplers
RandReal        meme::RANDREAL;
RandInt         meme::RANDINT;

// Local Helpers
FILE*           meme::STD_OUT;
FILE*           meme::STD_ERR;

/**
 * Entry point of the application
 * 
 * @param argc number of program arguments accessible in argv
 * @param argv array of arguments
 * @return int program succces
 */
int main(int argc, char *argv[]) {

    // Initialise MPI functionality
    // int mpi_size, mpi_rank;
    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    cout << setprecision(18);

    // Parse arguments
    args::load_args(argc, argv);

    cout << "Seed: " << meme::SEED << endl;
    RandInt ri = RandInt(meme::SEED);
    RandReal rr = RandReal(meme::SEED);
    RandInt::RANDINT = &ri;
    RandReal::RANDREAL = &rr;
    Model::FORMAT = PrintType::PrintExcel;

    // Read Train and Test, Output for local copy
    DataSet train = DataSet(meme::TRAIN_FILE, meme::GPU);
    train.load();
    train.csv(meme::LOG_DIR+to_string(meme::SEED)+".Train.csv");
    DataSet test = DataSet(meme::TEST_FILE, meme::GPU);
    test.load();
    test.csv(meme::LOG_DIR+to_string(meme::SEED)+".Test.csv");

    // Approximate derivative
    /*
    if( meme::IN_DER == "app-fd" )
        train.compute_app_der(meme::MAX_DER_ORD);

    train.normalise();
    */

    // Copy IVs to Model
    for(size_t i = 0; i < DataSet::IVS.size(); i++)
        ModelType::IVS.push_back(DataSet::IVS[i]);

    // Create Population & run
    Population<ModelType> p(&train);
    p.run();

    // Evaluate without derviative information for return
    meme::MAX_DER_ORD = 0;
    train = DataSet(meme::TRAIN_FILE, meme::GPU);
    train.load();

    // Write Run.log
    ofstream log(meme::LOG_DIR+to_string(meme::SEED)+".Run.log");
    if (log.is_open()) {

        log << setprecision(meme::PREC);

        // Get results
        vector<size_t> all;
        double train_score = ModelType::OBJECTIVE(&p.best_soln, &train, all);
        double test_score = ModelType::OBJECTIVE(&p.best_soln, &test, all);
        
        // Log data
        log << "Seed,Score,Train MSE,Test MSE,Dur,Model" << endl;
        log << meme::SEED;
        log << "," << MemeticModel<DataType>::OBJECTIVE_NAME;
        log << "," << train_score;
        log << "," << test_score;
        log << "," << meme::RUN_TIME/1000.0;
        log << "," << p.best_soln << endl;

        // While we are here, inform the user
        cout << endl << endl;
        cout << "Finished after " << meme::RUN_TIME << " milliseconds on seed " << meme::SEED << endl;
        cout << "====================================================" << endl << endl;
        cout << "Best Model Found" << endl;
        cout << p.best_soln << endl << endl;
        Model::FORMAT = PrintType::PrintLatex;
        cout << p.best_soln << endl << endl;
        cout << " Train MSE: " << train_score << endl;
        cout << "  Test MSE: " << test_score << endl << endl;
        cout << "====================================================" << endl << endl;

    }
    
    // MPI_Finalize();     // Close out MPI

    // Write Training results
    ofstream train_log(meme::LOG_DIR+to_string(meme::SEED)+".Train.Predict.csv");
    if (train_log.is_open()) {

        train_log << setprecision(meme::PREC) << "y";
        for(size_t i = 0; i < DataSet::IVS.size(); i++)
            train_log << "," << DataSet::IVS[i];
        train_log << ",yd" << endl;

        for(size_t i = 0; i < train.get_count(); i++) {

            train_log << train.y[i];
            for(size_t j = 0; j < DataSet::IVS.size(); j++)                    
                train_log << ","  << train.samples[i][j];
            train_log << "," << p.root_agent->get_pocket().evaluate(train.samples[i]) << endl;
            
        }

    }

    // Write Testing results
    ofstream test_log(meme::LOG_DIR+to_string(meme::SEED)+".Test.Predict.csv");
    if (test_log.is_open()) {

        test_log << setprecision(meme::PREC) << "y";
        for(size_t i = 0; i < DataSet::IVS.size(); i++)
            test_log << "," << DataSet::IVS[i];
        test_log << ",yd" << endl;
            
        for(size_t i = 0; i < test.get_count(); i++) {
            
            test_log << test.y[i];
            for(size_t j = 0; j < DataSet::IVS.size(); j++)                    
                test_log << ","  << test.samples[i][j];
            test_log << "," << p.root_agent->get_pocket().evaluate(test.samples[i]) << endl;
            
        }

    }

    return EXIT_SUCCESS;
}
