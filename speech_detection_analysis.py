from header_imports import *

if __name__ == "__main__":
    
    if len(sys.argv) != 1:

        if sys.argv[1] == "model_building":
            speech__analysis_obj = speech_building(model_type = sys.argv[2])

        if sys.argv[1] == "model_training":
            speech_analysis_obj = speech_training(model_type = sys.argv[2])
        
        if sys.argv[1] == "model_compare":
            pass

