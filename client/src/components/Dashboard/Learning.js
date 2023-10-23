/* 
Component - Enables selection of ML model
Functionality
- supports workking of MLdropdown.js
*/

import React, { useState } from 'react';
import './LearningDropdown.css';

const LearningDropdown = ({ onSelect }) => {
  const [selectedOption, setSelectedOption] = useState('');

  const handleButtonClick = (optionValue) => {
    setSelectedOption(optionValue);
    onSelect(optionValue);
  };

  return (
      <center>
        <div className="griddropdown">

        <button
    className={`grid-option ${selectedOption === 'supervised' ? 'selected' : ''}`}
    onClick={() => {
        alert('Supervised Models button clicked!');
        handleButtonClick('supervised');
    }}
>
    Supervised Models
</button>
<button
    className={`grid-option ${selectedOption === 'activeLearning' ? 'selected' : ''}`}
    onClick={() => {
        alert('Active Learning button clicked!');
        handleButtonClick('activeLearning');
    }}
>
    Active Learning
</button>


        </div>
      </center>
  );
};

export default LearningDropdown;
