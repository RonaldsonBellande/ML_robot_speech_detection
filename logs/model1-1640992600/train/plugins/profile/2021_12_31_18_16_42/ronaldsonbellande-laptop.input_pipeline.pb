	1(?h??@1(?h??@!1(?h??@	/Ϯ????/Ϯ????!/Ϯ????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$1(?h??@c_??`???Ac|??,V@Yh͏??@*	??(\O"?@2F
Iterator::Model?ؗl<?@!'c???lV@)??j??@1?>ѫ?J@:Preprocessing2U
Iterator::Model::ParallelMapV2?;?????!??hBQ?A@)?;?????1??hBQ?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??J"? ??!?^@???!@)+?`??1?F??P!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)??0???!????
??)0?AC???1vk???D??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap!u;?ʫ?!SN??~???)?g???c??1 J?^????:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceP?}:3??!ك"Q???)P?}:3??1ك"Q???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipj?{?ԗ??!?????$@)y?JxB???1'?(%T??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????!???A-??)??????1???A-??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9.Ϯ????Ib??b??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	c_??`???c_??`???!c_??`???      ??!       "      ??!       *      ??!       2	c|??,V@c|??,V@!c|??,V@:      ??!       B      ??!       J	h͏??@h͏??@!h͏??@R      ??!       Z	h͏??@h͏??@!h͏??@b      ??!       JCPU_ONLYY.Ϯ????b qb??b??X@