import React, { useState } from 'react';
import Home from './components/Navbar/Home';
import ImportCSV from './components/Dashboard/ImportCSV';
import MLdropdown from './components/Dashboard/MLdropdown';
import Results from './components/Dashboard/Results';
import DependencyGraphs from './components/Dashboard/DependencyGraphs';
import ROI from './components/Dashboard/ROI_analysis';
import ProgressBar from './components/Dashboard/ProgressBar';
import Navbar from './components/Navbar/Navbar';
import './App.css';

function App() {
  const [showOtherPage, setShowOtherPage] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [theme, setTheme] = useState('day'); // 'day' or 'night' theme
  const steps = ['Import CSV', 'Select Model', 'View Results', 'Graphs', 'ROI Analysis'];
  const pages = [ImportCSV, MLdropdown, Results, DependencyGraphs, ROI];

  const handleNextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleDashboardClick = () => {
    setShowOtherPage(true);
    console.log('Dashboard clicked');
  };

  const handleHomeClick = () => {
    setShowOtherPage(false);
  };

  const handleAboutClick = () => {
    setShowOtherPage(false);
  };

  const handleContactClick = () => {
    setShowOtherPage(false);
  };

  const CurrentPage = pages[currentStep];

  const handleModelSelect = (selectedModel) => {
    console.log('Selected model:', selectedModel);
  };

  const handleLearningSelect = (selectedModel) => {
    console.log('Selected model:', selectedModel);
  };

  const toggleTheme = () => {
    setTheme(theme === 'day' ? 'night' : 'day');
  };

  return (
    <div className={`App ${theme}`}>
      <header className="App-header">
        <Navbar
          onDashboardClick={handleDashboardClick}
          onHomeClick={handleHomeClick}
          onAboutClick={handleAboutClick}
          onContactClick={handleContactClick}
        />
      </header>
      <div className="App-content">
        {!showOtherPage && <Home />}
        {showOtherPage && (
          <>
            <ProgressBar steps={steps} currentStep={currentStep} />
            <CurrentPage
              onModelSelect={handleModelSelect}
              onLearningSelect={handleLearningSelect}
            />
            <div className="navigation-buttons">
              {currentStep > 0 && (
                <button className="prev-button" onClick={handlePrevStep}>
                  Previous
                </button>
              )}
              {currentStep < steps.length - 1 && (
                <button className="next-button" onClick={handleNextStep}>
                  Next
                </button>
              )}
            </div>
          </>
        )}
        <div className="theme-switch" onClick={toggleTheme}>
          <div className={`theme-switch-label ${theme === 'night' ? 'dark' : ''}`}>
            {theme === 'day' ? 'Day' : 'Night'}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
