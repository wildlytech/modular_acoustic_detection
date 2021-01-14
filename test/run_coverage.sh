#!/bin/sh

cd ../

coverage erase

# coverage run -a -m balancing_dataset
# coverage run -a -m create_base_dataframe
# # coverage run -a -m find_sounds
# coverage run -a -m generating_embeddings
# coverage run -a -m xenocanto_to_dataframe -b "prinia socialis" -o data/birds/
# coverage run -a -m youtube_audioset

# coverage run -a -m augmentation.audio_mixing

# coverage run -a -m compression.compression
# coverage run -a -m compression.decompression
# coverage run -a -m compression.different_compressions_latency

# coverage run -a -m continous_predictions.batch_test_ftp_files
# coverage run -a -m continous_predictions.batch_test_offline

# # coverage run -a -m Dash_integration.annotation.audio_annotation
# # coverage run -a -m Dash_integration.device_report.app_device
# # coverage run -a -m Dash_integration.monitoring_alert.app_Wildly_Acoustic_Monitoring
# # coverage run -a -m Dash_integration.multipage_ui.dash_integrate

# coverage run -a -m data_preprocessing_cleaning.mp3_stereo_to_wav_mono
# coverage run -a -m data_preprocessing_cleaning.split_wav_file
# coverage run -a -m data_preprocessing_cleaning.copy_files_by_csv
# coverage run -a -m data_preprocessing_cleaning.separate_by_label
# coverage run -a -m data_preprocessing_cleaning.identifying_mislabelled_silence_audiofiles
# coverage run -a -m data_preprocessing_cleaning.seperating_different_sounds
# coverage run -a -m data_preprocessing_cleaning.visualize_dataframe

# coverage run -a -m get_data.data_preprocessing
# # coverage run -a -m get_data.download_soi
coverage run -a -m get_data.ioc_bird_taxonomy_scrape -o data/ioc_ontology_extension.json
# coverage run -a -m get_data.youtube_scrape
# coverage run -a -m get_data.download_all_sounds
# coverage run -a -m get_data.xenocanto_scrape

# coverage run -a -m goertzel_filter.balancing_dataset_goertzel
# coverage run -a -m goertzel_filter.goertzel_algorithm
# coverage run -a -m goertzel_filter.goertzel_detection_model
# coverage run -a -m goertzel_filter.goertzel_filter_components

# Binary relevance model
coverage run -a -m models.binary_relevance_model -model_cfg_json test/model_configs/binary_relevance_model/domesticVsWild.json
coverage run -a -m models.binary_relevance_model -model_cfg_json test/model_configs/binary_relevance_model/explosion.json
# coverage run -a -m predictions.binary_relevance_model.generate_before_predict_BR
# coverage run -a -m predictions.binary_relevance_model.get_predictions_on_dataframe
coverage run -a -m predictions.binary_relevance_model.get_results_binary_relevance \
            --predictions_cfg_json=test/prediction_configs/binary_relevance_prediction_config.json \
            --path_for_dataframe_with_features=diff_class_datasets/Datasets/pure/Domestic/pure_dom_9497.pkl \
            --save_misclassified_examples=test/out_misclassified/br \
            --path_to_save_prediction_csv=test/out_csv/br_pred.csv
# coverage run -a -m predictions.binary_relevance_model.model_function_binary_relevance
# coverage run -a -m predictions.binary_relevance_model.predict_on_wavfile_binary_relevance


coverage run -a -m models.multilabel_model -cfg_json test/model_configs/multilabel_model/multilabel_maxpool.json
coverage run -a -m predictions.multilabel_model.multilabel_pred \
            --predictions_cfg_json=test/prediction_configs/multilabel_prediction_config.json \
            --path_for_dataframe_with_features=diff_class_datasets/Datasets/pure/Domestic/pure_dom_9497.pkl \
            --save_misclassified_examples=test/out_misclassified/multilabel \
            --path_to_save_prediction_csv=test/out_csv/multilabel_pred.csv


# coverage run -a -m predictions.multilabel_model.mutlilabel_pred_on_wavfile

# coverage run -a -m models.binary_model

coverage report -m -i

cd test/
