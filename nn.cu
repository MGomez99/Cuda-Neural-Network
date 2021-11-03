#include "nn.h"

using namespace std;


void run(vector<vector<float>> train_images, vector<vector<float>> test_images)
{

    /*
    W1 is N_NEURONS * PIXELS
    B1 is 1 * 1024  
    W2 is 10 * 1024 [10 is number of output layer neurons]
    B2 is 1 * 10 
    train_images is N_IMAGES * 785
    */
   cout <<"running"<<endl;
    vector<vector<float>> W1v(N_NEURONS, vector<float>(PIXELS));
    fill_matrix(W1v);

    vector<vector<float>> B1v(N_NEURONS, vector<float>(1));
    fill_matrix(B1v);

    vector<vector<float>> W2v(10, vector<float>(N_NEURONS));
    fill_matrix(W2v);

    vector<vector<float>> B2v(10, vector<float>(1));
    fill_matrix(B2v);
    cout << "generating random values" << endl;
    float *W1 = flatten_vector(W1v, N_NEURONS, PIXELS);
    float *B1 = flatten_vector(B1v, N_NEURONS, 1);
    float *W2 = flatten_vector(W2v, 10, N_NEURONS);
    float *B2 = flatten_vector(B2v, 10, 1);

    cout << "Parameters Initiallized" << endl;
    assert(train_images.size() == N_IMAGES && train_images[0].size() == PIXELS + 1);
    assert(test_images.size() == N_TEST_IMAGES && test_images[0].size() == PIXELS + 1);
    float *test_X = flatten_vector(test_images, N_TEST_IMAGES, (PIXELS));
    float *test_X_WITH_SOLN = flatten_vector(test_images, N_TEST_IMAGES, (PIXELS + 1));

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        std::shuffle(begin(train_images), std::end(train_images), gen);
        float *X_WHOLE_WITH_SOLN = flatten_vector(train_images, N_IMAGES, (PIXELS + 1)); // get shuffled data with labels
        float *X_WHOLE = flatten_vector(train_images, N_IMAGES, PIXELS); // get shuffled data without labels
        // for back prop
        float *Ys_WHOLE = (float *)malloc(10 * N_IMAGES * sizeof(float)); // encode labels
        // X_T is size 785 x SIZE -> last row's value has the index we want to write as a '1'
        for (int i = 0; i < N_IMAGES; i++)
        {
            int index = index_flattened(X_WHOLE_WITH_SOLN, i, PIXELS, (PIXELS + 1));
            index_flattened(Ys_WHOLE, index, i, N_IMAGES) = 1.f; // encode the proper solution as a 1
            for (int j = 0; j < 10; j++)
            {
                if (j != index)
                    index_flattened(Ys_WHOLE, j, i, N_IMAGES) = 0.f; // incorrect choices are set as '0'
            }
        } 
        /* 
        Z1 is 1024 * BATCHSIZE
        A1 is 1024 * BATCHSIZE
        Z2 is 10 * 1024
        A2 is 10 * 1024
        */
        cout << "EPOCH: " << epoch <<" / "<< EPOCHS<< endl;
        float train_acc;
        train_acc = getAccuracy(X_WHOLE, X_WHOLE_WITH_SOLN, W1, W2, B1, B2, N_IMAGES);
        cout << "\tTrain Accuracy: " << train_acc << endl;
        float test_acc;
        test_acc = getAccuracy(test_X, test_X_WITH_SOLN, W1, W2, B1, B2, N_TEST_IMAGES);
        cout << "\tTest Accuracy: " << test_acc << endl;
        int num_batches = floorf(N_IMAGES / BATCHSIZE);
        for (int batch = 0; batch < num_batches; batch++)
        {
            if (batch % 50 == 0)
                cout << "\t\tBATCH = " << batch << " / " << num_batches;

            int start = batch * BATCHSIZE;
            int end = start+BATCHSIZE;
           // cout<<"making data"<<endl;
            if (batch % 50 == 0 && TIMER)
                cout << "...generating shuffled batch...";
            auto t1s = chrono::high_resolution_clock::now();
            assert(end - start == BATCHSIZE);
            assert(train_images[0].size() == (PIXELS + 1));
            assert(train_images.size() == N_IMAGES);
            float *X_WITH_SOLN = flatten_vector_range_rows(train_images, start, end, BATCHSIZE, (PIXELS + 1)); // get current batch with labels
            //cout<<"Made X with soln"<<endl;
            float *X = flatten_vector_range_rows(train_images, start, end, BATCHSIZE, PIXELS); // get current batch without labels
            //cout<<"Made X without Soln"<<endl;
            float *Ys = (float *)malloc(10 * BATCHSIZE * sizeof(float)); // Ys will be 10 * BATCHSIZE
            //cout<<"made data"<<endl;
            for (int i = 0; i < end - start; i++)
            {
                int index = index_flattened(X_WITH_SOLN, i, PIXELS, (PIXELS + 1));
                index_flattened(Ys, index, i, BATCHSIZE) = 1.f; // encode the proper solution as a 1
                for (int j = 0; j < 10; j++)
                {
                    if (j != index)
                        index_flattened(Ys, j, i, BATCHSIZE) = 0.f; // incorrect choices are set as '0'
                }
            }
            auto t2s = chrono::high_resolution_clock::now();
            auto time_tots = ((t2s - t1s).count()) / 1000000;
            if (batch % 50 == 0 && TIMER)
                cout << "...current batch generation took " << time_tots << " ms...starting training...";
            t1s = chrono::high_resolution_clock::now();
            //cout<<"starting algorithm..."<<endl;



            float *Z1 = (float *)malloc(N_NEURONS * BATCHSIZE * sizeof(float));
            float *A1 = (float *)malloc(N_NEURONS * BATCHSIZE * sizeof(float));
            float *Z2 = (float *)malloc(10 * BATCHSIZE * sizeof(float));
            float *A2 = (float *)malloc(10 * BATCHSIZE * sizeof(float));
            forwardpass(X, start, W1, W2, B1, B2, Z1, Z2, A1, A2, BATCHSIZE);
            // TODO dropout

            float *dW1 = (float *)malloc(N_NEURONS * PIXELS * sizeof(float));
            float *dB1 = (float *)malloc(1* N_NEURONS * sizeof(float));
            float *dW2 = (float *)malloc(10 * N_NEURONS * sizeof(float));
            float *dB2 = (float *)malloc(1 * 10 * sizeof(float));

            //BACKPROP AND UPDATE

            backprop(X_WHOLE, Ys, start, Z1, A1, Z2, A2, W1, B1, W2, B2, dW1, dW2, dB1, dB2, BATCHSIZE);
            update_params(W1, W2, B1, B2, dW1, dW2, dB1, dB2);
            free(dW1);
            free(dW2);
            free(dB1);
            free(dB2);
            free(Z1);
            free(Z2);
            free(A1);
            free(A2);
            free(X);
            free(X_WITH_SOLN);
            free(Ys);
            t2s = chrono::high_resolution_clock::now();
            time_tots = ((t2s - t1s).count()) / 1000000;
            if (batch % 50 == 0 && TIMER)
                cout << "...training and updating parameters took " << time_tots << " ms..." << endl;
        }
        free(X_WHOLE);
        free(X_WHOLE_WITH_SOLN);
        free(Ys_WHOLE);
    }
    free(W1);
    free(W2);
    free(B1);
    free(B2);
    free(test_X);
    free(test_X_WITH_SOLN);
}

int main()
{

    vector<vector<float>> train_images = readImages("train-images.idx3-ubyte");
    vector<vector<float>> test_images = readImages("t10k-images.idx3-ubyte");
    vector<float> train_labels = readLabels("train-labels.idx1-ubyte");
    vector<float> test_labels = readLabels("t10k-labels.idx1-ubyte");
    std::random_device rd;
    gen.seed(static_cast<long unsigned int>(chrono::high_resolution_clock::now().time_since_epoch().count()));
    // train_images is 60k rows x 784 cols
    // test_images is 10k rows x 784 cols
    transpose_2d(train_images);
    transpose_2d(test_images);
    train_images.push_back(train_labels);
    test_images.push_back(test_labels);

    transpose_2d(train_images);
    transpose_2d(test_images);
    // Each row is a specific sample, with the label at the last position (index = 784)
    // last row is the labels for the corresponding image
    d = uniform_real_distribution<float>(-.5f, .5f);
    // train_images is 60k rows x 784 cols again
    // test_images is 10k rows x 784 cols again
    assert(train_images.size() == N_IMAGES && train_images[0].size() == PIXELS + 1);
    assert(test_images.size() == N_TEST_IMAGES && test_images[0].size() + PIXELS + 1);
    cout << train_images.size() << " x " << train_images[0].size() << endl;

    // vector<vector<float>> test = {{11.47, 2, 3}, {8.97, 4, 5}, {13.677, 6, 7}, {13.53, 8, 9}};
    // int N = test.size();
    // int M = test[0].size();

    // for(int i = 0; i < N; i++){
    //     for(int j = 0; j < M; j++){
    //            cout<<test[i][j]<<"\t";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;
    // cout << endl;
    // int start = 0, end = 2;
    // float* result = flatten_vector_range_rows(test, start, end, end-start, M);
    // for (int i = 0; i < (end - start); i++)
    // {
    //     for(int j = 0; j < M; j++){
    //         cout<<index_flattened(result, i, j, M)<<"\t";
    //     }
    //     cout<<endl;
    // }

    // cout<<endl;
    // for(int i = 0; i < (end-start)*M; i++){
    //     cout << result[i] << "\t";
    // }
    // cout<<endl;
    run(train_images, test_images);
}
