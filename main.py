import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.97  # 程序最多只能占用指定gpu80%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config = config)
import numpy as np
import time
from psoCNN import psoCNN
import matplotlib.pyplot as plt
import matplotlib
import sys  # 需要引入的包


# # 以下为包装好的 Logger 类的定义
# class Logger(object):
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#         # self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass

# sys.stdout = Logger('out_log_003.txt')

if __name__ == '__main__':

    # sys.stdout = Logger('out_log_003.txt')
    dataset = 'WTD'

    number_runs = 10
    number_iterations = 10
    population_size = 10

    batch_size_pso = 32
    batch_size_full_training = 32

    epochs_pso = 30
    epochs_full_training = 30

    max_conv_output_channels = 128
    max_fully_connected_neurons = 150

    min_layer = 3
    max_layer = 6

    # Probability of each layer type (should sum to 1)
    probability_convolution = 0.5
    probability_pooling = 0.4
    probability_fully_connected = 0.1

    max_conv_kernel_size = 5  ##设置成了【3，5，7】

    Cg = 0.5  ##加速因子？
    dropout = 0.2

    #####   run the algorithm   #####
    results_path = './results/' + dataset + "/"

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    all_gBest_metrics = np.zeros(number_runs)
    runs_time = []
    all_gbest_par = []
    best_gBest_acc = 0

    for i in range(number_runs):
        print("Run number: " + str(i))
        start_time = time.time()
        ##初始化和GA的过程
        pso = psoCNN(dataset=dataset, n_iter=number_iterations, pop_size=population_size,
                     batch_size=batch_size_pso, epochs=epochs_pso, min_layer=min_layer, max_layer=max_layer,
                     conv_prob=probability_convolution, pool_prob=probability_pooling,
                     fc_prob=probability_fully_connected, max_conv_kernel=max_conv_kernel_size,
                     max_out_ch=max_conv_output_channels, max_fc_neurons=max_fully_connected_neurons,
                     dropout_rate=dropout)
        ##PSO的过程
        pso.fit(Cg=Cg, dropout_rate=dropout)

        print(pso.gBest_acc)

        # Plot current gBest
        matplotlib.use('Agg')
        plt.plot(pso.gBest_acc)
        plt.xlabel("Iteration")
        plt.ylabel("gBest acc")
        plt.savefig(results_path + "gBest-iter" + str(i) + ".png")
        plt.close()

        print('gBest architecture: ')
        print(pso.gBest)
        print(pso.gBest.layers)

        np.save(results_path + "gBest_inter" + str(i) + "_acc_history.npy", pso.gBest_acc)

        np.save(results_path + "gBest_iter" + str(i) + "_test_acc_history.npy", pso.gBest_test_acc)

        end_time = time.time()

        running_time = end_time - start_time

        runs_time.append(running_time)

        # Fully train the gBest model found
        n_parameters = pso.fit_gBest(batch_size=batch_size_full_training, epochs=epochs_full_training,
                                     dropout_rate=dropout)
        all_gbest_par.append(n_parameters)

        # Evaluate the fully trained gBest model
        gBest_metrics = pso.evaluate_gBest(batch_size=batch_size_full_training)
        
        if gBest_metrics >= best_gBest_acc:
            best_gBest_acc = gBest_metrics

            # Save best gBest model
            best_gBest_yaml = pso.gBest.model.to_json()

            with open(results_path + "best-gBest-model.json", "w") as yaml_file:
                yaml_file.write(best_gBest_yaml)

            # Save best gBest model weights to HDF5 file
            pso.gBest.model.save_weights(results_path + "best-gBest-weights.h5")
        
        pso.gBest.model_delete()
        
        all_gBest_metrics[i] = gBest_metrics

        print("This run took: " + str(running_time) + " seconds.")

        # Compute mean accuracy of all runs
        all_gBest_mean_metrics = np.mean(all_gBest_metrics, axis=0)

        np.save(results_path + "/time_to_run.npy", runs_time)

        # Save all gBest metrics
        np.save(results_path + "/all_gBest_metrics.npy", all_gBest_metrics)

        # Save results in a text file
        output_str = "All gBest number of parameters: " + str(all_gbest_par) + "\n"
        output_str = output_str + "All gBest test accuracies: " + str(all_gBest_metrics) + "\n"
        output_str = output_str + "All running times: " + str(runs_time) + "\n"
        #output_str = output_str + "Mean loss of all runs: " + str(all_gBest_mean_metrics[0]) + "\n"
        output_str = output_str + "Mean accuracy of all runs: " + str(all_gBest_mean_metrics) + "\n"

        print(output_str)

        with open(results_path + "/final_results.txt", "w") as f:
            try:
                print(output_str, file=f)
            except SyntaxError:
                print >> f, output_str
