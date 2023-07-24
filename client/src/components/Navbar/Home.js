import React from 'react';
import './Home.css'

const Home = ({}) => {




  return (
  <div className='home'>
      <div className='title'>AutoROI</div>
      <div className='textData' style={{marginBottom:'30px'}}>
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore
        magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
        consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
        pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est
        laborum."
      </div>


<div class="rounded-corner" style={{justifyContent: 'center'}}>
  <iframe src="https://drive.google.com/file/d/12n8RsGXgeDG5ZXGwWdjzRhigtvvmtLBD/preview" width="1150" height="650"></iframe>
</div>
      <div className='headings'>Tool Architecture</div>
      <div style={{display: 'flex', justifyContent: 'center'}}>
  <img src="./roi.jpg" width="1100" height="600" alt="ROI Image"/>
</div>
<div style={{ width: '800px', height: '600px', overflow: 'hidden',backgroundColor: 'white' }}>
      <iframe
        src="./pdf1.pdf#toolbar=0"
        title="PDF Viewer"
        style={{ border: 'none', width: '100%', height: '100%', transform: 'scale(0.5)', transformOrigin: 'top left' , background:'transparent'}}
      />
  </div>

  </div>
  );
};

export default Home;
