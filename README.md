# Local training pipeline for "adam_bc / BrianChip v1"

This is the local training pipeline (based on Keras / TensorFlow) for your Edge Impulse project [adam_bc / BrianChip v1](http://localhost:4800/studio/204136) (http://localhost:4800/studio/204136). Use it to train your model locally or run experiments. Once you're done with experimentation you can push the model back into Edge Impulse, and retrain from there.

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Open a command prompt or terminal window.
3. Build the container:

    ```
    $ docker build -t custom-block-204136 .
    ```

4. Train your model:

    **macOS, Linux**

    ```
    $ docker run --rm -v $PWD:/scripts custom-block-204136 --data-directory data --out-directory out
    ```

    **Windows**

    ```
    $ docker run --rm -v "%cd%":/scripts custom-block-204136 --data-directory data --out-directory out
    ```

5. This will write your model (in TFLite, Saved Model and H5 format) to the `out/` directory.

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

## Pushing this block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation).
2. Make sure you run the training pipeline once locally - as we download some things on demand. This is not possible in Edge Impulse as your block runs without a connection to the outside world.
3. If you require extra dependencies you need to add them to `requirements.txt` (or call `poetry install` again in `Dockerfile`).
4. Open a command prompt or terminal window.
5. Push the block to Edge Impulse via:

    ```
    $ edge-impulse-blocks push
    ```

6. Depending on the data your block operates on, you can add it via:
    * Object Detection: **Create impulse > Add learning block > Object Detection (Images)**, then select the block via 'Choose a different model' on the 'Object detection' page.
    * Image classification: **Create impulse > Add learning block > Transfer learning (Images)**, then select the block via 'Choose a different model' on the 'Transfer learning' page.
    * Audio classification: **Create impulse > Add learning block > Transfer Learning (Keyword Spotting)**, then select the block via 'Choose a different model' on the 'Transfer learning' page.
    * Other (classification): **Create impulse > Add learning block > Custom classification**, then select the block via 'Choose a different model' on the 'Machine learning' page.
    * Other (regression): **Create impulse > Add learning block > Custom regression**, then select the block via 'Choose a different model' on the 'Regression' page.

### Testing this block (as an Edge Impulse developer)

1. Build the container (see above).
2. Go to an organization, **Custom blocks > Machine learning > Add new Machine learning block**.
3. Set 'Docker container' to `custom-block-204136`.
4. Add the block under **Create impulse**.

This is a lot faster than building through the CLI, as you bypass Kaniko.
