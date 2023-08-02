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


<div class="rounded-corner" style={{justifyContent: 'center', alignContent:'center', marginLeft:'100px'}}>
  <iframe src="https://drive.google.com/file/d/12n8RsGXgeDG5ZXGwWdjzRhigtvvmtLBD/preview" width="1000" height="600" ></iframe>
</div>
      <div className='headings' style={{marginTop:'50px'}}>Tool Architecture</div>
      <div style={{display: 'flex', justifyContent: 'center'}}>
      <img src="./roi.jpg" width="1100" height="600" alt="ROI Image"/>
</div>
<div className='headings'>Research Work</div>
<div style={{ display: 'flex', overflowX: 'auto', width: '1000px', alignContent: 'center', marginLeft: '100px', zIndex: 1  }}>
  <div style={{ position: 'relative', width: '500px', height: '400px', overflow: 'hidden', backgroundColor: 'white', marginRight: '20px' }}>
    <iframe
      src="./pdf1.pdf#toolbar=0"
      title="PDF Viewer"
      style={{ border: 'none', width: '100%', height: '100%', background: 'transparent',transform: 'scale(0.7)' }}
    />
    <a href="./pdf1.pdf#toolbar=0" target="_blank" style={{ position: 'absolute', bottom: '10px', right: '10px', width: '30px', height: '30px', background: 'black', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', textDecoration: 'none', color: 'white' }}>
      &#8599; 
    </a>
  </div>
  <div style={{ position: 'relative', width: '500px', height: '400px', overflow: 'hidden', backgroundColor: 'white' }}>
    <iframe
      src="./pdf1.pdf#toolbar=0"
      title="PDF Viewer"
      style={{ border: 'none', width: '100%', height: '100%', background: 'transparent',transform: 'scale(0.7)'}}
    />
    <a href="./pdf1.pdf#toolbar=0" target="_blank" style={{ position: 'absolute', bottom: '10px', right: '10px', width: '30px', height: '30px', background: 'black', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', textDecoration: 'none', color: 'white' }}>
      &#8599;
    </a>
  </div>
</div>

 {/* second row for pdf */}
<div style={{ display: 'flex', overflowX: 'auto', width: '1000px', alignContent: 'center', marginLeft: '100px', marginTop:'0px',zIndex: 1 }}>
  <div style={{ position: 'relative', width: '500px', height: '400px', overflow: 'hidden', backgroundColor: 'white', marginRight: '20px' }}>
    <iframe
      src="./pdf1.pdf#toolbar=0"
      title="PDF Viewer"
      style={{ border: 'none', width: '100%', height: '100%', background: 'transparent' ,transform: 'scale(0.7)'}}
    />
    <a href="./pdf1.pdf#toolbar=0" target="_blank" style={{ position: 'absolute', bottom: '10px', right: '10px', width: '30px', height: '30px', background: 'black', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', textDecoration: 'none', color: 'white' }}>
      &#8599; 
    </a>
  </div>
  <div style={{ position: 'relative', width: '500px', height: '400px', overflow: 'hidden', backgroundColor: 'white' }}>
    <iframe
      src="./pdf1.pdf#toolbar=0"
      title="PDF Viewer"
      style={{ border: 'none', width: '100%', height: '100%', background: 'transparent', transform: 'scale(0.7)' }}
    />
    <a href="./pdf1.pdf#toolbar=0" target="_blank" style={{ position: 'absolute', bottom: '10px', right: '10px', width: '30px', height: '30px', background: 'black', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', textDecoration: 'none', color: 'white' }}>
      &#8599;
    </a>
  </div>
</div>



</div>
  );
};

export default Home;
