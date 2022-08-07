import "./App.css";
import Dashboard from "./components/Dashboard";
import { BrowserRouter as Router, Route, Switch } from "react-router-dom";
function App() {
  return (
    <Router>
      <Switch>
        <Route path="/" exact component={Dashboard} />
        <Route path="/alpha" exact component={Dashboard} />
        <Route path="/beta" exact component={Dashboard} />
        <Route path="/gamma" exact component={Dashboard} />
      </Switch>
    </Router>
  );
}

export default App;
