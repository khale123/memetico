

/**
 * @file
 * @author andy@impv.au
 * @version 1.0
 * @brief Header for global variables and functions
 * 
 */

#ifndef MEMETICO_ARGS_H
#define MEMETICO_ARGS_H

#include <string>
#include <memetico/globals.h>
#include <memetico/model_base/model_meme.h>
#include <memetico/data/data_set.h>
#include <memetico/optimise/objective.h>
#include <memetico/optimise/local_search.h>
#include <memetico/models/cont_frac_dd.h>
#include <memetico/models/branch_cont_frac_dd.h>
#include <memetico/population/pop.h>
#include <memetico/global_types.h>

namespace args {

    stringstream    args_out;

    bool arg_exists(char** begin, char** end, string short_option, string long_option) {

        if(find(begin, end, short_option) != end || find(begin, end, long_option) != end) {
            args_out << "    \"" << short_option << "\"" << endl;
            return true;
        }

        return false;
    }

    string arg_value(char** begin, char** end, string short_option, string long_option) {
        
        string ret;

        // Check for short string e.g. "-h" rather than "-help"
        char ** itr = find(begin, end, short_option);
        if (itr != end && ++itr != end)
            ret = string(*itr);
        
        // Check for long string e.g.  "-help"
        itr = find(begin, end, long_option);
        if (itr != end && ++itr != end)
            ret = string(*itr);

        // Build debug for VSC launch.json
        args_out << "    \"" << short_option << "\", \"" << ret << "\"," << endl;

        return ret;

    }

    void arg_seed(int argc, char * argv[]) {

        // Seed
        string arg_string = arg_value(argv, argv+argc, "-s", "--seed");
        if(arg_string != "")
            meme::SEED = stol(arg_string);
        else  {
            RandInt seed_init = RandInt();
            meme::SEED = seed_init(1, numeric_limits<int>::max());
        }
        meme::RANDINT = RandInt(meme::SEED);
        meme::RANDREAL = RandReal(meme::SEED);

    }

    void arg_log(int argc, char * argv[]) {

        // Log Directory
        string arg_string = arg_value(argv, argv+argc, string("-lt"), string("--log-to"));
        if(arg_string != "") {
            meme::LOG_DIR = arg_string;
            if( meme::LOG_DIR.back() != '/' )
                meme::LOG_DIR += "/";

            // Change STD out if the log dir is specified. If not, use normal stdout / stderr
            meme::STD_OUT = freopen( (meme::LOG_DIR + to_string(meme::SEED) + ".Std.log").c_str(), "a", stdout);
            meme::STD_ERR = freopen( (meme::LOG_DIR + to_string(meme::SEED) + ".Err.log").c_str(), "a", stderr);

        }
    }

    void arg_local_search(int argc, char * argv[]) {
        // Local Search function
        string arg_string = arg_value(argv, argv+argc, "-ls", "--local-search");
        if( arg_string == "cnm" || arg_string == "")    MemeticModel<DataType>::LOCAL_SEARCH = local_search::custom_nelder_mead_redo<MemeticModel<DataType>>;
        if( arg_string == "cnms" )                      MemeticModel<DataType>::LOCAL_SEARCH = local_search::custom_nelder_mead_alg4<MemeticModel<DataType>>;  
    }

    void arg_objective(int argc, char * argv[]) {
        // Objective function
        string arg_string = arg_value(argv, argv+argc, "-o", "--objective");
        MemeticModel<DataType>::OBJECTIVE_NAME = arg_string;
        if( arg_string == "")                                   MemeticModel<DataType>::OBJECTIVE_NAME = "mse";
        if( MemeticModel<DataType>::OBJECTIVE_NAME == "mse" )   MemeticModel<DataType>::OBJECTIVE = objective::mse<DataType>;
        if( MemeticModel<DataType>::OBJECTIVE_NAME == "mae" )   MemeticModel<DataType>::OBJECTIVE = objective::mae<DataType>;
        if( MemeticModel<DataType>::OBJECTIVE_NAME == "rmse" )   MemeticModel<DataType>::OBJECTIVE = objective::rmse<DataType>;
        if( MemeticModel<DataType>::OBJECTIVE_NAME == "mape" )   MemeticModel<DataType>::OBJECTIVE = objective::mape<DataType>;
        if( MemeticModel<DataType>::OBJECTIVE_NAME == "pcor" )   MemeticModel<DataType>::OBJECTIVE = objective::p_cor<DataType>;
        if( MemeticModel<DataType>::OBJECTIVE_NAME == "scor" )   MemeticModel<DataType>::OBJECTIVE = objective::s_cor<DataType>;

        // With the derviative, it may make more sense to have an independent flag for 'to include derivative information'
        // which is a flag used in the different objective funtions, e.g. mse, mae, etc. to include the information 
        if( MemeticModel<DataType>::OBJECTIVE_NAME == "mse_der" )  MemeticModel<DataType>::OBJECTIVE = objective::mse_der<DataType>;
        if( MemeticModel<DataType>::OBJECTIVE_NAME == "mse_der_multiV" )  MemeticModel<DataType>::OBJECTIVE = objective::mse_der_multiV<DataType>;
    
    }

    void arg_dynamic_depth(int argc, char * argv[]) {
        // Dynamic Depth Type
        string arg_string = arg_value(argv, argv+argc, "-dd", "--dynamic-depth");
        if(arg_string == "none" || arg_string == "")    meme::DYNAMIC_DEPTH_TYPE = DynamicNone;
        if(arg_string == "adp")                         meme::DYNAMIC_DEPTH_TYPE = DynamicAdaptive;
        if(arg_string == "adp-mu")                      meme::DYNAMIC_DEPTH_TYPE = DynamicAdaptiveMutation;
        if(arg_string == "adp-rnd")                     meme::DYNAMIC_DEPTH_TYPE = DynamicRandom;
    }

    void arg_diversity(int argc, char * argv[]) {
        // Diversity Type
        string arg_string = arg_value(argv, argv+argc, "-dm", "--diversity-method");
        if(arg_string == "none" || arg_string == "")    Population<ModelType>::DIVERSITY_TYPE = DiversityNone;
        if(arg_string == "every")                       Population<ModelType>::DIVERSITY_TYPE = DiversityEvery;
        if(arg_string == "stale")                       Population<ModelType>::DIVERSITY_TYPE = DiversityStale;
        if(arg_string == "stale-ext")                   Population<ModelType>::DIVERSITY_TYPE = DiversityStaleExtended;
    }

    void arg_help(int argc, char * argv[]) {
     
        if(arg_exists(argv, argv+argc, "-h", "--help")) {
            
            cout << 
            
            R"(
                    Memetico      
                

                        USAGE:

                            ./meme-cpp/bin/main;

                        REQUIRED:

                            -lt --log-to <filedir>          Directory for output files
                                    Defaults to <repo>/out/

                            -t --train <filepath>           Training full filepath


                        OPTIONAL:

                            -cu --cuda                      Execute with cuda GPU optimisation

                            -d --delta                      Penalty for the number of parameters within the solution (Real Number 0.0 <= d <= 1.0)
                                                            Expressed in the solution fitness. When no parameters are used Fitness = Error           
                                                            When parameters are used
                                                                Factor = 1+used_params()*Penalty
                                                                Fitness = Error*Factor
                                                            Defaults to 0.35

                            -dc --diversity-count           Number of solutions to replace in 'stale' and 'every' --diversity-method's

                            -dd --dynamic-depth             Method of dynamic depth
                                                                none: hardset depth for the solution, -f
                                                                adp: randomise new solutions with pocket depth +- 1
                                                                adp-mu: as per adp, but during mutation re-randomise depth +- 1
                                                                rnd: randomise new solutions uniform at random between 0 and -f inclusive
                                                                
                            -dm --diversity-method          Method to replace similar solutions each generation
                                                            Compares all pocket and current solutions based on differences in MSE scores
                                                            Process performs a full local search after renewing
                                                                none (default): no replacement method
                                                                every: Every generation replace the --diversity-count most similar solutions
                                                                stale: Replace --diversity-count most similar solutions on every --stale 
                                                                stale-ext: As per stale but every two consecutive --stale's, 50% of the solutions are replaced
                                                            Defaults to 0

                            -e --epsilon                    Error threshold for stopping condition
                                                            Defaults to 0

                            -id --inder                     Input derivative mode (exact, app-fd, or app-multiV)
                                                            Defaults to exact

                            -ifr --index-first-repetition   Index first repetition, 1 core = 1 repetition
                                                            Defaults to 0


                            -f --fracdepth <integer>        Depth of continued fraction
                                                            Defaults to 4

                            -g --gens                       Number of generations to evolve the memetic population
                                                            Defaults to 200
                            
                            -ld --local-data                Percentage of data to use in local search between 0 and 1
                                                            Defaults to 1

                            -ls --local-search              Local Search method
                                                            Available Options: 
                                                                cnm
                                                            Defaults to "cnm

                            -mdo, --max-der-ord             Maximum derivative order (compute derivatives up to this order)
                                                            Defaults to 1

                            -nn --num-neighbors             Number of neighbors to consider in multiV derivative
                                                            Defaults to 16

                            -norm --normalization           Normalize errors in mse_der_multiV
                                                            Defaults to false

                            -p --problem                    Problem name

                            -o --objective                  Objective function identifier to drive learning
                                                            Available Options: 
                                                                mse, 
                                                                // nsme, 
                                                                // cw,
                                                                // aic
                                                            Defaults to "mse"

                            -mr --mutate-rate               Percentage of solutions that undergo mutation each generation between 0 and 1
                                                            Defaults to 0.2

                            -mt --max-time                  Number of seconds after which the best solution is returned

                            -s --seed <integer>             Reproduction seed
                                                            Defaults to random integer between 1, numerical_limit<int>::max()

                            -st --stale                     Stale count that triggers renew of the roots current solution

                            

                            -T --Test <filepath>            Test data file for interpolation
                                                            Defaults to training filepath from -t


                )";
        }
    }
    
    void load_args(int argc, char * argv[]) {
   
        args_out << "==================" << endl;
        args_out << "Arguments" << endl;
        args_out << "==================" << endl;
        args_out << "Command: " << argv[0] << endl;
        args_out << "\"args\": [" << endl;
        
        string arg_string;

        // Cuda 
        if( arg_exists(argv, argv+argc, string("-cu"), string("--cuda")) )
            meme::GPU = true;

        // Train
        arg_string = arg_value(argv, argv+argc, string("-t"), string("--train"));
        if(arg_string != "")    meme::TRAIN_FILE = arg_string;

        // Test
        arg_string = arg_value(argv, argv+argc, "-T", "--Test");
        if(arg_string != "")    meme::TEST_FILE = arg_string;
        else                    meme::TEST_FILE = meme::TRAIN_FILE;
        
        // Mutation Rate
        arg_string = arg_value(argv, argv+argc, "-mr", "--mutate-rate");
        if(arg_string != "")    meme::MUTATE_RATE = stod(arg_string);
        
        // Penalty
        arg_string = arg_value(argv, argv+argc, "-d", "--delta");
        if(arg_string != "")    meme::PENALTY = stod(arg_string);

        // Generations
        arg_string = arg_value(argv, argv+argc, "-g", "--gens");
        if(arg_string != "")    meme::GENERATIONS = stoi(arg_string);

        // Local Search Data
        arg_string = arg_value(argv, argv+argc, "-ld", "--local-data");
        if(arg_string != "")    meme::LOCAL_SEARCH_DATA_PCT = stod(arg_string);

        // Stale Count
        arg_string = arg_value(argv, argv+argc, "-st", "--stale");
        if(arg_string != "")        meme::STALE_RESET = stoi(arg_string);
        
        // Max time
        arg_string = arg_value(argv, argv+argc, "-mt", "--max-time");
        if(arg_string != "")        meme::MAX_TIME = stoi(arg_string);

        // Error epsilon
        arg_string = arg_value(argv, argv+argc, "-e", "--error");
        if(arg_string != "")        meme::EPSILON = stod(arg_string);

        arg_seed(argc, argv);
        arg_log(argc, argv);
        arg_local_search(argc, argv);
        arg_objective(argc, argv);

        // CFR Specific

        // Depth
        arg_string = arg_value(argv, argv+argc, "-f", "--fracdepth");
        if(arg_string != "")        meme::DEPTH = stoi(arg_string);
        
        // Diversity Count
        arg_string = arg_value(argv, argv+argc, "-dc", "--diversity-count");
        if(arg_string != "")        meme::DIVERSITY_COUNT = stoi(arg_string);

        arg_dynamic_depth(argc,argv);
        arg_diversity(argc,argv);

        //// Derivative considerations

        // Index First Repetition (IFR)
        arg_string = arg_value(argv, argv+argc, string("-ifr"), string("--index-first-repetition"));
        if(arg_string != "")    meme::IFR = stoi(arg_string);

        // Input derivative mode
        arg_string = arg_value(argv, argv+argc, string("-id"), string("--inder"));
        if(arg_string != "")    meme::IN_DER = arg_string;

        // Maximum derivative order
        arg_string = arg_value(argv, argv+argc, "-mdo", "--max-der-ord");
        if(arg_string != "")        meme::MAX_DER_ORD = stoi(arg_string);

        // MultiV Added

        // Number of neighbors
        arg_string = arg_value(argv, argv+argc, "-nn", "--num-neighbors");
        if(arg_string != "")    meme::NUM_NEIGHBORS = stoi(arg_string);

        // Normalization flag
        if(arg_exists(argv, argv+argc, "-norm", "--normalization"))
            meme::NORMALIZATION_FLAG = true;
               
        // Master log
        meme::master_log = ofstream(meme::LOG_DIR+to_string(meme::SEED)+".Master.log");

        // Output arguments in VSCode launch.json format
        args_out << "]," << endl;
        cout << args_out.str();
        
    }
}

#endif
