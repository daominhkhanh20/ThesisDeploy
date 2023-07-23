import React, { useState } from 'react';

const SearchBar = ({ handleSearch }) => {
  const [searchTerm, setSearchTerm] = useState('');

  const handleChange = (event) => {
    setSearchTerm(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    handleSearch(searchTerm);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={searchTerm}
        onChange={handleChange}
        placeholder="Search..."
      />
      <button type="submit">Search</button>
    </form>
  );
};

const SearchResult = ({ results }) => {
  return (
    <div className="result-container">
      {results.map((result) => (
        <div key={result.id} className="result-item">
          <a href={result.link}>{result.title}</a>
          <p>{result.description}</p>
        </div>
      ))}
    </div>
  );
};

const App = () => {
  const [searchResults, setSearchResults] = useState([]);

  const handleSearch = (searchTerm) => {
    // Simulated API call or search logic
    const results = [
      { id: 1, title: 'Result 1', description: 'Description for Result 1', link: '#' },
      { id: 2, title: 'Result 2', description: 'Description for Result 2', link: '#' },
      { id: 3, title: 'Result 3', description: 'Description for Result 3', link: '#' },
    ];

    setSearchResults(results);
  };

  return (
    <div className="app">
      <SearchBar handleSearch={handleSearch} />
      <SearchResult results={searchResults} />
    </div>
  );
};

export default App;