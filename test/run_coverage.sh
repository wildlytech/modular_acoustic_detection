#!/bin/sh

cd ../

coverage erase

##coverage run -a -m get_data.download_soi #Takes a long time, introduce a test input argument
coverage run -a -m get_data.ioc_bird_taxonomy_scrape -o data/ioc_ontology_extension.json
coverage run -a -m get_data.youtube_scrape  #Script is broken, page layout has changed
##coverage run -a -m get_data.download_all_sounds #Takes a long time, introduce a test input argument 

coverage run -a -m get_data.xenocanto_scrape -bird_species "rhinoptilus bitorquatus" -path_to_save_audio_files ./test/bird_sounds_xc/

cp -r ./test/bird_sounds_xc/rhinoptilus_bitorquatus/*.* ./test/test_mp3/

coverage run -a -m data_preprocessing_cleaning.mp3_stereo_to_wav_mono -path_to_save_wav_files test/test_wav/ -input_mp3_path test/test_mp3/
coverage run -a -m data_preprocessing_cleaning.split_wav_file -path_for_wavfiles test/test_wav/ -path_to_write_chunks test/wav_10sec/

coverage run -a -m generating_embeddings -wav_file test/wav_10sec/ -path_to_write_embeddings test/test_embeddings
coverage run -a -m create_base_dataframe -path_for_saved_embeddings test/test_embeddings -path_to_write_dataframe test/test_dataframe.pkl

# coverage run -a -m balancing_dataset -- Update user gude. How does it work?
# coverage run -a -m find_sounds -- Update user gude. How does it work?
# coverage run -a -m youtube_audioset -- Update user gude. How does it work?

coverage run -a -m xenocanto_to_dataframe -b "rhinoptilus bitorquatus" -o data/birds/

# coverage run -a -m augmentation.audio_mixing -- Update user gude. How does it work?

coverage run -a -m compression.compression -path_to_original_audio_files test/test_wav/ -path_to_compressed_audio_files test/compressed/ -codec_type aac 
coverage run -a -m compression.compression -path_to_original_audio_files test/test_wav/ -path_to_compressed_audio_files test/compressed/ -codec_type mp3
coverage run -a -m compression.compression -path_to_original_audio_files test/test_wav/ -path_to_compressed_audio_files test/compressed/ -codec_type flac
coverage run -a -m compression.compression -path_to_original_audio_files test/test_wav/ -path_to_compressed_audio_files test/compressed/ -codec_type ac3
coverage run -a -m compression.compression -path_to_original_audio_files test/test_wav/ -path_to_compressed_audio_files test/compressed/ -codec_type mp2

coverage run -a -m compression.decompression -path_to_compressed_audio_files test/compressed/ -path_to_decompressed_audio_files test/decompressed/ -codec_type aac
coverage run -a -m compression.decompression -path_to_compressed_audio_files test/compressed/ -path_to_decompressed_audio_files test/decompressed/ -codec_type mp3
coverage run -a -m compression.decompression -path_to_compressed_audio_files test/compressed/ -path_to_decompressed_audio_files test/decompressed/ -codec_type flac
coverage run -a -m compression.decompression -path_to_compressed_audio_files test/compressed/ -path_to_decompressed_audio_files test/decompressed/ -codec_type ac3
coverage run -a -m compression.decompression -path_to_compressed_audio_files test/compressed/ -path_to_decompressed_audio_files test/decompressed/ -codec_type mp2

coverage run -a -m compression.different_compressions_latency

#coverage run -a -m continous_predictions.batch_test_ftp_files -ftp_folder_path /home/user-u0xzU/ftp_test/bnp/ -csv_filename bnp_predictions_FTP.csv ## Ad login deails and is broken! 
coverage run -a -m continous_predictions.batch_test_offline -local_folder_path test/wav_10sec/ -csv_filename test/test_predictions_offline.csv

# coverage run -a -m continous_predictions.generate_before_predict_BR
# coverage run -a -m continous_predictions.model_function_binary_relevance

# # coverage run -a -m Dash_integration.annotation.audio_annotation
# # coverage run -a -m Dash_integration.device_report.app_device
# # coverage run -a -m Dash_integration.monitoring_alert.app_Wildly_Acoustic_Monitoring
# # coverage run -a -m Dash_integration.multipage_ui.dash_integrate

# coverage run -a -m data_preprocessing_cleaning.copy_files_by_csv
# coverage run -a -m data_preprocessing_cleaning.separate_by_label
# coverage run -a -m data_preprocessing_cleaning.identifying_mislabelled_silence_audiofiles
# coverage run -a -m data_preprocessing_cleaning.seperating_different_sounds

coverage run -a -m data_preprocessing_cleaning.visualize_dataframe -f test/dataframe_with_labels.pkl 
coverage run -a -m get_data.data_preprocessing -annotation_file test/test_annotations.csv -path_for_saved_embeddings test/test_embeddings/ -path_to_save_dataframe test/test_dataframe.pkl

# coverage run -a -m goertzel_filter.balancing_dataset_goertzel
# coverage run -a -m goertzel_filter.goertzel_algorithm
# coverage run -a -m goertzel_filter.goertzel_detection_model
# coverage run -a -m goertzel_filter.goertzel_filter_components

# Binary relevance model
##coverage run -a -m models.binary_relevance_model -model_cfg_json test/model_configs/binary_relevance_model/domesticVsWild.json
##coverage run -a -m models.binary_relevance_model -model_cfg_json test/model_configs/binary_relevance_model/explosion.json
# coverage run -a -m predictions.binary_relevance_model.generate_before_predict_BR
# coverage run -a -m predictions.binary_relevance_model.get_predictions_on_dataframe
##coverage run -a -m predictions.binary_relevance_model.get_results_binary_relevance \
##            --predictions_cfg_json=test/prediction_configs/binary_relevance_prediction_config.json \
##            --path_for_dataframe_with_features=diff_class_datasets/Datasets/pure/Domestic/pure_dom_9497.pkl \
##            --save_misclassified_examples=test/out_misclassified/br \
##            --path_to_save_prediction_csv=test/out_csv/br_pred.csv
# coverage run -a -m predictions.binary_relevance_model.model_function_binary_relevance
# coverage run -a -m predictions.binary_relevance_model.predict_on_wavfile_binary_relevance


##coverage run -a -m models.multilabel_model -cfg_json test/model_configs/multilabel_model/multilabel_maxpool.json
##coverage run -a -m predictions.multilabel_model.multilabel_pred \
##            --predictions_cfg_json=test/prediction_configs/multilabel_prediction_config.json \
##            --path_for_dataframe_with_features=diff_class_datasets/Datasets/pure/Domestic/pure_dom_9497.pkl \
##            --save_misclassified_examples=test/out_misclassified/multilabel \
##            --path_to_save_prediction_csv=test/out_csv/multilabel_pred.csv


# coverage run -a -m predictions.multilabel_model.mutlilabel_pred_on_wavfile

# coverage run -a -m models.binary_model

coverage report -m -i

cd test/
