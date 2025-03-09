import React, { useState } from 'react';
import { Search, FileText, PieChart, Settings, MessageSquare, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Download, Star, RotateCw, HelpCircle, List } from 'lucide-react';

const ModernSECAnalyzer = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [selectedRating, setSelectedRating] = useState(null);
  
  // Dummy chat messages
  const [messages, setMessages] = useState([
    { 
      id: 1, 
      type: 'user', 
      text: 'What are Apple\'s risk factors in their 2023 10-K?' 
    },
    { 
      id: 2, 
      type: 'assistant', 
      text: 'Based on the Apple Inc. 2023 10-K filing, key risk factors include:\n\n1. Global economic conditions that have adversely impacted Apple\'s results of operations\n2. Supply chain disruptions affecting production and distribution\n3. Intense competition in all business areas\n4. Rapid technological changes requiring continuous innovation\n5. Intellectual property challenges and litigation risks\n6. International operations subject to complex laws and regulations\n7. Data protection and cybersecurity concerns\n\nThese factors could materially impact the company in the future as discussed in Item 7 and Item 1A of the filing.' 
    }
  ]);
  
  // Dummy sample queries
  const sampleQueries = [
    "What are Apple's risk factors in their 2023 10-K?",
    "Tell me about Microsoft's revenue for 2022",
    "What does Tesla say about competition in their latest 10-K?",
    "Summarize Amazon's growth strategy for 2023",
    "Compare Google and Meta's R&D spending"
  ];
  
  // Function to handle sending a new message
  const handleSendMessage = () => {
    // In a real app, this would send the message to your backend
    console.log("Message sent");
  };
  
  // Toggle sidebar
  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className={`bg-navy-900 text-white transition-all duration-300 ${sidebarCollapsed ? 'w-16' : 'w-64'}`}>
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          {!sidebarCollapsed && <h1 className="text-xl font-semibold">SEC Analyzer</h1>}
          <button onClick={toggleSidebar} className="p-1 rounded hover:bg-gray-700">
            {sidebarCollapsed ? <ChevronRight /> : <ChevronLeft />}
          </button>
        </div>
        
        <nav className="mt-6">
          <div className={`flex items-center px-4 py-3 ${activeTab === 'search' ? 'bg-blue-700' : 'hover:bg-gray-700'}`} 
               onClick={() => setActiveTab('search')}>
            <Search className="h-5 w-5" />
            {!sidebarCollapsed && <span className="ml-3">Search Filings</span>}
          </div>
          
          <div className={`flex items-center px-4 py-3 ${activeTab === 'chat' ? 'bg-blue-700' : 'hover:bg-gray-700'}`} 
               onClick={() => setActiveTab('chat')}>
            <MessageSquare className="h-5 w-5" />
            {!sidebarCollapsed && <span className="ml-3">Chat Interface</span>}
          </div>
          
          <div className={`flex items-center px-4 py-3 ${activeTab === 'analytics' ? 'bg-blue-700' : 'hover:bg-gray-700'}`} 
               onClick={() => setActiveTab('analytics')}>
            <PieChart className="h-5 w-5" />
            {!sidebarCollapsed && <span className="ml-3">Analytics</span>}
          </div>
          
          <div className={`flex items-center px-4 py-3 ${activeTab === 'documents' ? 'bg-blue-700' : 'hover:bg-gray-700'}`} 
               onClick={() => setActiveTab('documents')}>
            <FileText className="h-5 w-5" />
            {!sidebarCollapsed && <span className="ml-3">My Documents</span>}
          </div>
          
          <div className={`flex items-center px-4 py-3 ${activeTab === 'settings' ? 'bg-blue-700' : 'hover:bg-gray-700'}`} 
               onClick={() => setActiveTab('settings')}>
            <Settings className="h-5 w-5" />
            {!sidebarCollapsed && <span className="ml-3">Settings</span>}
          </div>
          
          {!sidebarCollapsed && 
            <div className="px-4 py-6 mt-6 border-t border-gray-700">
              <h3 className="text-sm uppercase text-gray-400 font-medium">Recent Companies</h3>
              <ul className="mt-3 space-y-2">
                <li className="text-sm hover:text-blue-300 cursor-pointer">Apple Inc. (AAPL)</li>
                <li className="text-sm hover:text-blue-300 cursor-pointer">Microsoft Corp. (MSFT)</li>
                <li className="text-sm hover:text-blue-300 cursor-pointer">Tesla Inc. (TSLA)</li>
                <li className="text-sm hover:text-blue-300 cursor-pointer">Amazon.com (AMZN)</li>
              </ul>
            </div>
          }
        </nav>
      </div>
      
      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-sm p-4 flex items-center justify-between">
          <div className="flex items-center">
            <h2 className="text-xl font-semibold text-gray-800">SEC Filing Analyzer</h2>
          </div>
          <div className="flex items-center space-x-4">
            <button className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded">
              <HelpCircle className="h-5 w-5" />
            </button>
            <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center text-white">
              <span className="text-sm font-medium">JS</span>
            </div>
          </div>
        </header>
        
        {/* Main content area */}
        <main className="flex-1 flex overflow-hidden">
          {/* Primary content area */}
          <div className={`flex-1 flex flex-col ${activeTab === 'chat' ? 'max-w-md md:max-w-lg' : 'max-w-full'} border-r border-gray-200`}>
            {activeTab === 'chat' && (
              <>
                <div className="bg-white p-4 border-b">
                  <h3 className="text-lg font-medium text-gray-800">Chat Interface</h3>
                  <p className="text-sm text-gray-500">Ask questions about SEC filings</p>
                </div>
                <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
                  {messages.map((message) => (
                    <div key={message.id} className={`mb-4 ${message.type === 'user' ? 'text-right' : ''}`}>
                      <div className={`inline-block p-3 rounded-lg max-w-xs md:max-w-md ${
                        message.type === 'user' 
                          ? 'bg-blue-600 text-white rounded-br-none' 
                          : 'bg-white text-gray-800 rounded-bl-none shadow'
                      }`}>
                        <p className="whitespace-pre-line">{message.text}</p>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="p-4 bg-white border-t">
                  <div className="mb-3 space-y-1">
                    <p className="text-xs font-medium text-gray-500">SAMPLE QUERIES</p>
                    <div className="flex flex-wrap gap-2">
                      {sampleQueries.map((query, index) => (
                        <button 
                          key={index}
                          className="text-xs px-2 py-1 bg-gray-100 rounded-full hover:bg-gray-200 text-gray-700 whitespace-nowrap overflow-hidden text-ellipsis max-w-full"
                        >
                          {query}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-end gap-2">
                    <input
                      type="text"
                      placeholder="Ask about a company's SEC filing..."
                      className="flex-1 border rounded-lg py-2 px-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <button 
                      onClick={handleSendMessage}
                      className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700"
                    >
                      Send
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
          
          {/* PDF Viewer (only shown in Chat view) */}
          {activeTab === 'chat' && (
            <div className="flex-1 flex flex-col bg-gray-100">
              <div className="bg-white p-4 border-b flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-medium text-gray-800">Filing Document</h3>
                  <p className="text-sm text-gray-500">Apple Inc. 10-K (2023)</p>
                </div>
                <div className="flex items-center space-x-2">
                  <button className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded">
                    <ZoomOut className="h-5 w-5" />
                  </button>
                  <button className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded">
                    <ZoomIn className="h-5 w-5" />
                  </button>
                  <button className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded">
                    <Download className="h-5 w-5" />
                  </button>
                </div>
              </div>
              
              <div className="flex-1 overflow-auto bg-gray-800 p-4 flex justify-center">
                <div className="bg-white shadow-lg max-w-2xl w-full">
                  {/* This would be your actual PDF.js viewer */}
                  <div className="p-8">
                    <div className="text-center py-4">
                      <div className="mx-auto w-16 h-16 mb-4">
                        <svg viewBox="0 0 1024 1024" className="w-full h-full">
                          <path d="M747.88 725.57h-471.9c-6.29 0-10.05-3.78-10.05-10.06V308.49c0-6.28 3.76-10.05 10.05-10.05h471.9c6.28 0 10.05 3.77 10.05 10.05v407.02c0 6.28-3.77 10.06-10.05 10.06" fill="#FDFDFD"/>
                          <path d="M271.53 300.59v412.6h480.94v-412.6H271.53z m5.3 5.29h470.35v402.02H276.83V305.88z" fill="#E6E6E6"/>
                          <path d="M512 410.8c38.73 0 70.27-31.54 70.27-70.27 0-38.73-31.54-70.27-70.27-70.27-38.72 0-70.26 31.54-70.26 70.27 0 38.73 31.54 70.27 70.26 70.27" fill="#E74C3C"/>
                          <path d="M354.57 634.05h314.71v15.91H354.57zM354.62 677.89h314.72v15.91H354.62zM354.62 721.73h314.72v15.91H354.62z" fill="#BDC3C7"/>
                          <path d="M511.98 232.73c38.73 0 70.27 31.54 70.27 70.27 0 38.73-31.54 70.27-70.27 70.27-38.72 0-70.26-31.54-70.26-70.27 0-38.73 31.54-70.27 70.26-70.27m0-20c-49.76 0-90.26 40.5-90.26 90.27 0 49.76 40.5 90.27 90.26 90.27 49.77 0 90.27-40.5 90.27-90.27 0-49.76-40.5-90.27-90.27-90.27" fill="#E74C3C"/>
                        </svg>
                      </div>
                      <h2 className="text-lg font-semibold mb-1">ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)</h2>
                      <h3 className="text-md mb-6">For the fiscal year ended September 30, 2023</h3>
                    </div>
                    
                    <div className="text-center mb-6">
                      <div className="mx-auto w-12 h-12 mb-2">
                        <svg viewBox="0 0 1024 1024" className="w-full h-full">
                          <path d="M708.1 429.5c0.2-3.4 0.4-6.8 0.4-10.2 0-103.8-84.9-188-189.2-188-92.7 0-170.1 66.8-185.6 154.9-14.4-7.5-30.7-11.8-47.9-11.8-57.2 0-103.5 46.5-103.5 103.8 0 0.5 0 0.9 0.1 1.4-55.5 17.8-93.2 69.8-93.2 121.8 0 66.8 54.7 132.6 122.8 132.6h466.4c68.1 0 123.6-55.8 123.6-124.4-0.1-71.6-55.5-126.1-124-130.1z" fill="#E6F7FF"/>
                        </svg>
                      </div>
                      <h2 className="text-xl font-bold mb-1">Apple Inc.</h2>
                      <h3 className="text-sm text-gray-500 mb-4">(Exact name of registrant as specified in its charter)</h3>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-8 mb-8">
                      <div className="text-center p-4 border border-gray-200 rounded">
                        <h4 className="text-xs text-gray-500 mb-1">State of incorporation</h4>
                        <p className="font-medium">California</p>
                      </div>
                      <div className="text-center p-4 border border-gray-200 rounded">
                        <h4 className="text-xs text-gray-500 mb-1">I.R.S. Employer Identification No.</h4>
                        <p className="font-medium">94-2404110</p>
                      </div>
                    </div>
                    
                    <div className="text-center p-4 border border-gray-200 rounded mb-8">
                      <h4 className="text-xs text-gray-500 mb-1">Address of principal executive offices</h4>
                      <p className="font-medium">One Apple Park Way</p>
                      <p className="font-medium">Cupertino, California 95014</p>
                      <p className="text-sm mt-2">(408) 996-1010</p>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Feedback section */}
              <div className="bg-white p-4 border-t">
                <h4 className="text-sm font-medium text-gray-700 mb-2">How was this analysis?</h4>
                <div className="flex items-center space-x-4 mb-3">
                  <label className="flex items-center cursor-pointer">
                    <input 
                      type="radio" 
                      name="rating" 
                      className="mr-2" 
                      checked={selectedRating === 'incorrect'} 
                      onChange={() => setSelectedRating('incorrect')}
                    />
                    <span className="text-sm">Incorrect</span>
                  </label>
                  <label className="flex items-center cursor-pointer">
                    <input 
                      type="radio" 
                      name="rating" 
                      className="mr-2" 
                      checked={selectedRating === 'partial'} 
                      onChange={() => setSelectedRating('partial')}
                    />
                    <span className="text-sm">Partially Correct</span>
                  </label>
                  <label className="flex items-center cursor-pointer">
                    <input 
                      type="radio" 
                      name="rating" 
                      className="mr-2" 
                      checked={selectedRating === 'spot_on'} 
                      onChange={() => setSelectedRating('spot_on')}
                    />
                    <span className="text-sm">Spot On</span>
                  </label>
                </div>
                <textarea 
                  className="w-full border rounded p-2 text-sm" 
                  rows="2" 
                  placeholder="Provide additional feedback on this analysis (optional)..."
                ></textarea>
                <button className="mt-2 bg-blue-600 text-white px-3 py-1 text-sm rounded hover:bg-blue-700">
                  Submit Feedback
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default ModernSECAnalyzer;