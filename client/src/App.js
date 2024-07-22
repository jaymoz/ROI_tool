import React, { useState, useEffect } from 'react';
import Home from './components/Navbar/Home';
import Contact from './components/Navbar/Contact/Contact';
import ImportCSV from './components/Dashboard/ImportCSV';
import DependencyGraphs from './components/Dashboard/DependencyGraphs';
import ROI from './components/Dashboard/ROI_analysis';
import Navbar from './components/Navbar/Navbar';
import { Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard/Dashboard';
import './App.css';

// const TRACKING_ID = "UA-280046123-1";
// ReactGA.initialize(TRACKING_ID);
function App() {
  const [showOtherPage, setShowOtherPage] = useState(false);
  const [showContactPage, setShowContactPage] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [theme, setTheme] = useState('day'); // 'day' or 'night' theme
  const steps = ['Import CSV', 'Select Model', 'View Results',  'ROI Analysis', 'Graphs - Work in progress'];
  const pages = [ImportCSV, ROI, DependencyGraphs];


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
    setShowContactPage(false);
    console.log('Dashboard clicked');
  };

  const handleHomeClick = () => {
    setShowOtherPage(false);
    setShowContactPage(false);
  };


  const handleContactClick = () => {
    setShowOtherPage(false);
    setShowContactPage(true);
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

      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/dashboard/*" element={<Dashboard />} />        
      </Routes>
    </div>

      // <div className={`App ${theme}`}>
      //   <header className="App-header">
      //     <Navbar
      //         onDashboardClick={handleDashboardClick}
      //         onHomeClick={handleHomeClick}
      //         onContactClick={handleContactClick}
      //     />
      //   </header>
      //   <div className="App-content">
      //     {!showOtherPage && showContactPage && <Contact/>}
      //     {!showOtherPage && !showContactPage && <Home />}
      //     {showOtherPage && (
      //         <>
      //           <ProgressBar steps={steps} currentStep={currentStep} />
      //           <CurrentPage
      //               onModelSelect={handleModelSelect}
      //               onLearningSelect={handleLearningSelect}
      //           />
      //           <div className="navigation-buttons">
      //             {currentStep > 0 && (
      //                 <button className="prev-button" onClick={handlePrevStep}>
      //                   Previous
      //                 </button>
      //             )}
      //             {currentStep < steps.length - 1 && (
      //                 <button className="next-button" onClick={handleNextStep}>
      //                   Next
      //                 </button>
      //             )}
      //           </div>
      //         </>
      //     )}
      //     <div className="theme-switch" onClick={toggleTheme}>
      //       <div className={`theme-switch-label ${theme === 'night' ? 'dark' : ''}`}>
      //         {theme === 'day' ? 'Day' : 'Night'}
      //       </div>
      //     </div>
      //   </div>
      // </div>
  );
}

export default App;
