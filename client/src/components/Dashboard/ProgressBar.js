import React from 'react';
import './ProgressBar.css';

const ProgressBar = ({ steps, currentStep, onNext }) => {
  const stepWidth = 100 / (steps.length - 1);
  const circleCircumference = 2 * Math.PI * 70; // Radius is assumed as 70, change it if needed
  const progressPercentage = (currentStep / (steps.length - 1)) * 100;
  const strokeDasharrayValue = (progressPercentage / 100) * circleCircumference;

  return (
    <>
    <div className="progress-bar-container">
      <div className="progress-bar">
        {steps.map((step, index) => (
          <div
            className={`step ${index <= currentStep ? 'active' : ''}`}
            key={index}
            style={{ width: `${stepWidth}%` }}
          >
            {index !== steps.length - 1 && <div className="step-line"></div>}
            <div className="step-circle">
              <span className="step-label">{step}</span>
            </div>
            
          </div>
        ))}
        
      </div>
      <div className="circular-progress-bar">
      
        <svg className="progressbar__svg">
          <circle
            cx="80"
            cy="80"
            r="70"
            className="progressbar__svg-circle"
          ></circle>
          <circle
            cx="80"
            cy="80"
            r="70"
            className="progressbar__svg-circle-progress"
            style={{ strokeDasharray: `${strokeDasharrayValue} ${circleCircumference}` }}
          ></circle>
        </svg>
        <span className="progressbar__text">{`${Math.round(progressPercentage)}%`}</span>
        <p className="completion-status">Completion Status</p>
      </div>

    </div>
    <div className="horizontal-line"></div>
    </>
    
  );
};

export default ProgressBar;
