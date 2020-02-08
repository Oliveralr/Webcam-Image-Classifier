const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let net;

async function app() {
    console.log('Loading Mobile Net...');

    //Load the model
    net = await mobilenet.load();
    console.log('Succesfully loaded model!');

    //This creates an object which could capture image from 
    //the web camera as a Tensor.
    const webcam = await tf.data.webcam(webcamElement);

    //Reads an image from the webcam and associates with a specifi class index
    const addExample = async classId => {
        const img = await webcam.capture();
        //Get MobileNet activation 'conv_pred'
        const activation = net.infer(img, 'conv_preds');
        //Pass throug classificator
        classifier.addExample(activation, classId);
        img.dispose();
    };
    
    //Button functionality
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));

    while(true){
        if(classifier.getNumClasses() > 0){
            const img = await webcam.capture();
            const activation = net.infer(img, 'conv_preds');

            const result = await classifier.predictClass(activation);
            const classes = ['A', 'B', 'C'];

            document.getElementById('console').innerText = `
                prediction: ${classes[result.label]}\n
                probability: ${result.confidences[result.label]}
            `;
            img.dispose();
        }
        //Give some breathing by waiting for the next animation frame
        await tf.nextFrame();
    }
}

app();