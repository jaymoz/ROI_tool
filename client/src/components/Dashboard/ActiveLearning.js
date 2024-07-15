import React, { useState } from 'react';
import axios from 'axios';
import './ActiveLearning.css';

const ActiveLearning = () => {
  const [threshold, setThreshold] = useState(0.60);
  const [maxIterations, setMaxIterations] = useState(3);
  const [resampling, setResampling] = useState('under_sampling');
  const [classifier, setClassifier] = useState('RF');
  const [samplingType, setSamplingType] = useState('leastConfidence');
  const [testSize, setTestSize] = useState(0.2);
  const [manualAnnotationsCount, setManualAnnotationsCount] = useState(12);
  const [comments] = useState('-RF_LC-12-12');
  const [fileContent, setFileContent] = useState(''); // new state to hold file content
  const [iterationIndex, setIterationIndex] = useState(1); // start at iteration 1
  const [displayContent, setDisplayContent] = useState(''); // content to display
  const [manuallyAnnotatedData, setManuallyAnnotatedData] = useState(''); // new state to hold manually annotated data
  const [moreContentVisible, setMoreContentVisible] = useState(false); // state to control visibility of "more" content


  const handleSubmit = async () => {
    try {
        const response = await axios.post(
            "https://roibackend.shaktilab.org/activeLearning1", 
            {
                threshold: threshold,
                max_iterations: maxIterations,
                resampling: resampling,
                classifier: classifier,
                sampling_type: samplingType,
                test_size: testSize,
                manual_annotations_count: manualAnnotationsCount,
                comments: comments
            },
            {
                headers: {
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
                }
            }
        );

        setFileContent(response.data.fileContent)
        setIterationIndex(0);
    } catch (error) {
        console.error("Error posting data:", error);
        // Handle the error, e.g., show an error message to the user
    }
};

  const handleIterationChange = async (direction) => {
    let newIndex = iterationIndex;

    if (direction === 'next' && iterationIndex < maxIterations) {
      newIndex = iterationIndex + 1;
    } else if (direction === 'prev' && iterationIndex > 1) {
      newIndex = iterationIndex - 1;
    }

    const delimiterExists = fileContent.includes(`Iteration : ${newIndex}`);

    let newFileContent = fileContent;
    if (direction === 'next' && !delimiterExists) {
      // If "Next" is pressed and delimiter for the new iteration doesn't exist, update the file content.
      try {
          const response = await axios.post(
              "https://roibackend.shaktilab.org/next",
              {},  // No data to send with the POST request, keep it an empty object
              {
                  headers: {
                      "Access-Control-Allow-Origin": "*",
                      "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE,PATCH,OPTIONS",
                      "X-Requested-With": "XMLHttpRequest"  // Required by some versions of cors-anywhere
                  }
              }
          );
          newFileContent = response.data.fileContent;
          setFileContent(newFileContent);
      } catch (error) {
          console.error("Error posting data:", error);
          // Handle the error, e.g., show an error message to the user
      }
  }
  
    // Extract iteration content from either the existing file content or the new one fetched from the backend.
    const newIterationContent = extractIteration(newFileContent, newIndex);
    setDisplayContent(newIterationContent);

    setIterationIndex(newIndex);
    setMoreContentVisible(false);
  };

  const extractManuallyAnnotatedData = (fileContent, iteration) => {
    const lines = fileContent.split('\n');
    let count = 0;
    let indexStart = -1;
    let indexEnd = -1;

    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes('Manually Annotated Combinations')) {
        count += 1;
        if (count === iteration) {
          indexStart = i;
        }
      }
      if (lines[i].includes('Merging Newly Labelled Data Samples')) {
        if (count === iteration) {
          indexEnd = i;
          break;
        }
      }
    }
    return lines.slice(indexStart , indexEnd).join('\n');
  };

  const extractIteration = (fileContent, iteration) => {
    const lines = fileContent.split('\n');
    let count = 0;
    let index = -1;

    // Loop through the lines until we find the nth occurrence of 'Analysis DataFrame :'
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes('Analysis DataFrame :')) {
        count += 1;
        if (count === iteration) {
          index = i;
          break;
        }
      }
    }

    console.log(index);  // logging the found index

    // Return 25 lines above 'Analysis DataFrame :' up to 2 lines above it
    // Make sure the slice start isn't less than 0
    return lines.slice(Math.max(0, index - 23), index - 2).join('\n');
  };



  const handleMoreButtonClick = () => {
    // Extract the manually annotated data for the current iteration
    const newData = extractManuallyAnnotatedData(fileContent, iterationIndex);
    // Update the state with the new data
    setManuallyAnnotatedData(newData);
    // Toggle visibility of "more" content
    setMoreContentVisible(!moreContentVisible);
  };

  return (
    <div className="active-learning-container">
    <form className="active-learning">
      <section className="form-inputs">
      <div className="input-container">
        <label className="input-label">
          Threshold: {threshold}
        </label>
        <input
          className="input-slider"
          type="range"
          min="0"
          max="1"
          step="0.01" // Changed from 0.1 to 0.01
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />
      </div>
      <div className="input-container">
        <label className="input-label">
          Max Iterations: {maxIterations}
        </label>
        <input
          className="input-slider"
          type="range"
          min="1"
          max="10"
          value={maxIterations}
          onChange={(e) => setMaxIterations(Number(e.target.value))}
        />
      </div>
      <div className="input-container">
        <label className="input-label">
          Test Size: {testSize}
        </label>
        <input
          className="input-slider"
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={testSize}
          onChange={(e) => setTestSize(Number(e.target.value))}
        />
      </div>
      <div className="input-container">
        <label className="input-label">Sampling Type:</label>
        <select
          value={samplingType}
          onChange={(e) => setSamplingType(e.target.value)}
        >
          <option value="leastConfidence">Least Confidence</option>
          <option value="minMargin">Min Margin</option>
          <option value="entropy">Entropy</option>
        </select>
      </div>
      <div className="input-container">
        <label className="input-label">Classifier Type:</label>
        <select
          value={classifier}
          onChange={(e) => setClassifier(e.target.value)}>
          <option value="RF">Random Forest</option>
          <option value="NB">Naive Bayes</option>
          <option value="SVM">Support Vector Machine</option>
          <option value="ensemble">Ensemble</option>
        </select>
      </div>
      <div className="input-container">
        <label className="input-label">Resampling:</label>
        <select
        value={resampling}
        onChange={(e) => setResampling(e.target.value)}>
          <option value="over_sampling">Over Sampling</option>
          <option value="under_sampling">Under Sampling</option>
        </select>
      </div>

      <div className="input-container">
        <label className="input-label">Manual Annotations Count:{manualAnnotationsCount}</label>
        <input
          className="input-slider"
          type="range"
          min="1"
          max="100"
          value={manualAnnotationsCount}
          onChange={(e) => setManualAnnotationsCount(Number(e.target.value))}
        />
      </div>
      </section>


      <div className="input-container">
        <label className="input-label-iteration"> {classifier} Current Iteration: {iterationIndex}</label>
      </div>

      <section className="iteration-navigation">
        {iterationIndex > 1 && <button type="button" onClick={() => handleIterationChange('prev')}>Previous Iteration</button>}
        {iterationIndex < maxIterations && <button type="button" onClick={() => handleIterationChange('next')}>Next Iteration</button>}
      </section>


      <div className="file-content">
        {iterationIndex > 0 && <pre>{displayContent}</pre>}
      </div>

      <button type="button" onClick={handleSubmit}>Submit</button>
        <button type="button" onClick={handleMoreButtonClick}>More</button>

      </form>
      <div className={`annotated-content ${moreContentVisible ? 'active' : ''}`}>
        {iterationIndex > 0 && <pre>{manuallyAnnotatedData}</pre>}
      </div>
    </div>
  );
};

export default ActiveLearning;
