import React from 'react';
import './Home.css'

const Home = ({}) => {

  return (
      <div className='home'>
        <div className='title'>AROhI</div>

        <div className='textData' style={{marginBottom:'30px'}}>
          Most Machine Learning (ML) solutions are evaluated only based on accuracy. However, ML algorithms for a given problem generally require higher computational resources than classical algorithms for the same problem. In addition, these algorithms also need large training datasets, which tends to become an effort and cost-intensive aspect. Given the above issues, this research provided a more profound outlook on the ROI (Return on Investment) of data analytics and the tradeoff between cost and benefit for an ML selection. Essentially, we tried to answer the question, “How much data analytics is enough?” and provided a rough estimate of the cost-benefit of using an ML-based solution. This research work proposed looking beyond accuracy measures and considering Return on Investment (ROI) as an additional criterion to evaluate conventional and complex ML models (such as Semisupervised and Deep Learning (DL)).
        </div>

        <video className='home-video' controls>
            <source src="/roi_demo.mp4" type="video/mp4" />
            Your browser does not support the video tag.
        </video>
      </div>
  );
};

export default Home;
