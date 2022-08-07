import {
    Backdrop,
    Box,
    Button, ButtonGroup,
    CircularProgress,
    Grid,
    IconButton,
    makeStyles,
    Paper, Snackbar, Tab, Tabs,
    TextField, Tooltip,
    Typography
} from '@material-ui/core';
import Autocomplete from '@material-ui/lab/Autocomplete';
import AddIcon from '@material-ui/icons/Add';
import RemoveIcon from '@material-ui/icons/Remove';
import DeleteIcon from '@material-ui/icons/Delete';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import * as React from 'react';
import {useContext, useEffect, useState} from 'react';
import {paper} from '../styles';
import TensorTable from './TensorTable';
import {useForm} from "react-hook-form";
import axios from 'axios';
import * as assert from "assert";
import PlayArrowIcon from "@material-ui/icons/PlayArrow";
import CancelIcon from "@material-ui/icons/Cancel";
import {TensorFlowOperations} from "../Constants";
import Preference from './Preference';
import {Alert, TabPanel} from "@material-ui/lab";
import { BaselineContext } from './Dashboard';
import CheckCircleIcon from '@material-ui/icons/CheckCircle';
import Stopwatch from "./Stopwatch";

const range = length => [...Array.from({length}).keys()]
const useStyles = makeStyles((theme) => ({
    paper: {
        height: '5rem',
        ...paper(theme)
    },
    button: {
        margin: theme.spacing(2),
    },
    textField: {
        margin: theme.spacing(2),
        height: '2rem',
    },
    backdrop: {
        zIndex: theme.zIndex.drawer + 1,
        color: '#fff',
    },
}));
let synthesizingValue = false;
let abortSessionId = null;

const ControlPanel = ({setValidations, allInputTensors, setAllInputTensors, allOutputTensors, setAllOutputTensors, synthesisSessionId, setSynthesisSessionId, convertTestCases, validations, setSynthesizedSolutions, preference, setPreference, synthesizing, setSynthesizing, selectedTestCase, setSelectedTestCase, selectedSolution, setErrorSnackbarOpen, setErrorSnackbarMessage}) => {
    synthesizingValue = synthesizing;
    const baseline = useContext(BaselineContext)
    const classes = useStyles();
    const [description, setDescription] = useState('');
    const [constants, setConstants] = useState('');
    const [synthesisTimeout, setSynthesisTimeout] = useState(300);
    const [solutionCount, setSolutionCount] = useState(3);
    const [useBackdrop, setUseBackdrop] = useState(false);
    const [lastSynthesisResult, setLastSynthesisResult] = useState(null);
    const [synthesisStartTimestamp, setSynthesisStartTimestamp] = useState(new Date());
    const [synthesisEndTimestamp, setSynthesisEndTimestamp] = useState(new Date());

    const convertPref2List = () =>{
        const desiredOperations = [];
        const undesiredOperations = [];
        for(let op in preference){
            if(preference[op].desired) desiredOperations.push(op);
            else undesiredOperations.push(op);
        }
        return {desiredOperations, undesiredOperations}
    }
    const {desiredOperations, undesiredOperations} = convertPref2List();
    const updatePreference = (newOpList, desired) => {
        const {desiredOperations, undesiredOperations} = convertPref2List()
        const oldOpList = (desired) ? desiredOperations : undesiredOperations;
        const isAdd = newOpList.length > oldOpList.length;
        const longerList = (newOpList.length > oldOpList.length) ? newOpList : oldOpList
        const newPreference = {...preference}
        let diffOp = undefined;
        oldOpList.sort()
        newOpList.sort();
        for(let index in longerList){
            if(oldOpList[index]!==newOpList[index]) {
                diffOp = longerList[index];
                break;
            };
        }
        if(isAdd){
            newPreference[diffOp] = {
                label: diffOp,
                desired
            };
        }
        else{
            delete newPreference[diffOp]
        }
        setPreference(newPreference)
    }
    const addPair = () => {
        setAllInputTensors([...allInputTensors, allInputTensors[selectedTestCase].map((tensor, index) => {
            return {
                index: index,
                shape: tensor.shape,
                stringValue: tensor.stringValue,
                value: tensor.value
            }
        })])
        setAllOutputTensors([...allOutputTensors, {
            index: 0,
            shape: allOutputTensors[selectedTestCase].shape,
            stringValue: allOutputTensors[selectedTestCase].stringValue,
            value: allOutputTensors[selectedTestCase].value
        }])
    }
    const removeSelectedPair = () => {
        const newAllInputTensors = [
            ...allInputTensors.slice(0, selectedTestCase),
            ...allInputTensors.slice(selectedTestCase + 1)
        ]
        setAllInputTensors(newAllInputTensors)
        const newAllOutputTensors = [
            ...allOutputTensors.slice(0, selectedTestCase),
            ...allOutputTensors.slice(selectedTestCase + 1)
        ]
        setAllOutputTensors(newAllOutputTensors)
    }
    const addInputTensor = (pair) => {
        if (pair >= allInputTensors.length) {
            console.error(`Attempted to add pair ${pair} tensor but only ${allInputTensors.length} pair(s) exists`)
            return
        }
        const newAllInputTensors = [...allInputTensors]
        const newTensor = {
            index: newAllInputTensors[pair].length,
            shape: null,
            stringValue: '',
            value: null
        }
        newAllInputTensors[pair] = [...newAllInputTensors[pair], newTensor]
        setAllInputTensors(newAllInputTensors)
    }
    const removeInputTensor = (pair, index) => {
        const newAllInputTensors = [...allInputTensors]
        if (newAllInputTensors[pair].length === 1) return;
        newAllInputTensors[pair].splice(index, 1)
        setAllInputTensors(newAllInputTensors)
    }
    const parseTensor = str => {
        try {
            const parsed = JSON.parse(str.replaceAll('True', '"True"').replaceAll('False', '"False"'));
            const shape = getTensorShape(parsed);
            if (shape === null) return [null, null];
            return [parsed, shape];
        } catch (e) {
            return [null, null];
        }
    }
    const getTensorShape = tensor => {
        if (Number.isFinite(tensor)) return [];
        if (tensor === 'True' || tensor === 'False') return [];
        if (!Array.isArray(tensor)) return null;
        if (tensor.length === 0) return null;
        let shape = getTensorShape(tensor[0]);
        if (shape === null) return null;
        for (let i = 1; i < tensor.length; i++) {
            if (!areArraysEqual(getTensorShape(tensor[i]), shape)) return null;
        }
        if (shape.length === 0) {
            // Terminal case
            return [tensor.length];
        }
        return [tensor.length, ...shape];
    }
    const areArraysEqual = (a, b) => {
        if (a === b) return true;
        if (a == null || b == null) return false;
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) return false;
        }
        return true;
    }
    const setInputTensorAt = (pair, index, tensor) => {
        const newAllInputTensors = [
            ...allInputTensors.slice(0, pair),
            [
                ...allInputTensors[pair].slice(0, index),
                tensor,
                ...allInputTensors[pair].slice(index + 1)
            ],
            ...allInputTensors.slice(pair + 1)
        ]
        setAllInputTensors(newAllInputTensors)
    }
    const setOutputTensorAt = (pair, tensor) => {
        const newAllOutputTensors = [...allOutputTensors]
        newAllOutputTensors[pair] = tensor
        setAllOutputTensors(newAllOutputTensors);
    }
    const onTensorTableUpdatedAt = (tensor, path, value) => {
        if (tensor.value === null) return;
        if (value.length >= 1) {
            let booleanValue = null;
            if (value[0].toLowerCase() === 't') {
                booleanValue = true;
            } else if (value[0].toLowerCase() === 'f') {
                booleanValue = false;
            }
            if (booleanValue !== null) {
                value = booleanValue ? 'True' : 'False';
            } else {
                value = Number(value);
                if (!Number.isFinite(value)) value = 0;
            }
        } else {
            value = 0;
        }
        let array = tensor.value;
        if (path.length === 2 && tensor.shape.length === 1) {
            // Edge case: For 2D vectors rendered in 2D table (See TensorTable.jsx)
            tensor.value[path[1]] = value;
        } else if (path.length === 1) {
            tensor.value[path[0]] = value;
        } else {
            let pointer = array;
            for (let i = 0; i < path.length - 1; i++) {
                pointer = pointer[path[i]];
                assert(Array.isArray(pointer));
            }
            pointer[path[path.length - 1]] = value;
        }
        tensor.stringValue = JSON.stringify(tensor.value).replaceAll('"True"', 'True').replaceAll('"False"', 'False')
        return tensor;
    }
    const onInputTensorValueUpdate = (pair, index, stringValue) => {
        const [parsed, shape] = parseTensor(stringValue);
        setInputTensorAt(pair, index, {index: index, shape: shape, stringValue: stringValue, value: parsed});
    }
    const onOutputTensorValueUpdate = (pair, stringValue) => {
        const [parsed, shape] = parseTensor(stringValue);
        setOutputTensorAt(pair, {shape: shape, stringValue: stringValue, value: parsed})
    }
    const getTensorHelperText = (tensor) => {
        if (tensor.stringValue === '') return null;
        if (tensor.shape === null) return 'Invalid tensor.'
        return `Shape: ${JSON.stringify(tensor.shape)}`;
    }
    const onSubmit = (data) => {
        console.log('Submitting.')
        console.log(data)
        console.log(preference)

        data.description = description
        data.constraints = constants
        data.sol_num = solutionCount
        data.timeout = synthesisTimeout

        if (allInputTensors.some(tensors => tensors.some(it => it.shape === null)) || allOutputTensors.some(it => it.shape === null)) {
            setErrorSnackbarMessage('Errors are found in the test cases. Please fix them before continuing.')
            setErrorSnackbarOpen(true);
            return;
        }
        if (data.description.trim() === '') {
            setErrorSnackbarMessage('You must enter the description for the tensor manipulation.')
            setErrorSnackbarOpen(true);
            return;
        }

        // Postprocess data
        let postData = Object.assign({}, data);
        postData = Object.assign(postData, convertTestCases());
        postData.timeout = data.timeout;
        postData.constraints = data.constraints ? JSON.stringify(data.constraints.split(',').map(str => Number(str.trim()))) : '[]';
        postData.desired_op = {};
        postData.undesired_op = {};
        for (let op in preference) {
            if (preference[op].desired) postData.desired_op[op] = '';
            else postData.undesired_op[op] = '';
        }
        postData.sol_num = data.sol_num;
        console.log(postData);

        setSynthesizingValue(true);
        setLastSynthesisResult(null);
        setSynthesisStartTimestamp(new Date());
        setValidations({})
        axios.post('/solve', postData)
            .then(res => {
                onReceiveSolve(res)
            })
            .catch(e => {
                console.log(e);
                setSynthesizingValue(false);
                setSynthesisEndTimestamp(new Date());
            });
    }
    const onReceiveSolve = (res) => {
        setSynthesisSessionId(res.data.session_id)
        console.log(`Session ID: ${res.data.session_id}`);

        setSynthesizedSolutions({});

        if (!synthesizingValue) {
            console.log('Not synthesizing. Returning.')
        } else {
            pollSynthesisResults(res.data.session_id);
        }
    }
    const pollSynthesisResults = (sessionId) => {
        if (!synthesizingValue) {
            console.log('Should abort synthesis. Returning.')
            setSynthesizingValue(false);
            return;
        }

        abortSessionId = sessionId
        axios.get(`/poll/${sessionId}`)
            .then(res => {
                console.log(res.data)
                const {graphs, solutions} = res.data;
                if (solutions[`solution0`] != null) {
                    setSynthesizedSolutions(solutions);
                    setLastSynthesisResult(res.data);
                }
                if (res.data.completed) {
                    if (solutions[`solution0`] == null) {
                        setErrorSnackbarMessage('No solutions found!');
                        setErrorSnackbarOpen(true);
                        setSynthesizedSolutions({});
                    }
                    setSynthesizingValue(false);
                    setSynthesisEndTimestamp(new Date());
                } else {
                    setTimeout(() => pollSynthesisResults(sessionId), 500);
                }
            })
            .catch(e => {
                console.log(e);
                setErrorSnackbarMessage('Failed to poll synthesizer data.')
                setErrorSnackbarOpen(true);
                setSynthesizingValue(false);
                setSynthesisEndTimestamp(new Date());
            })
    }
    const abortSynthesis = () => {
        if (!abortSessionId) return
        axios.get(`/abort/${abortSessionId}`)
            .then(abortSessionId = null)
            .catch(e => {
                console.log(e);
                setErrorSnackbarMessage('Synthesizer failed to abort.')
                setErrorSnackbarOpen(true);
            })
        setSynthesizingValue(false);
        setSynthesisEndTimestamp(new Date());
    }
    const setSynthesizingValue = v => {
        synthesizingValue = v;
        setSynthesizing(v);
    }

    return (
        <form>
            <Box style={{height: 'calc(100vh - 64px)', position: 'relative', display: 'flex', flexDirection: 'column'}}>
                <Box style={{flex: 1, overflowY: 'scroll'}}>
                    {!baseline &&<Box sx={{ borderBottom: 1, borderColor: 'divider', display: 'flex' }}>
                    <Tabs
                            style={{ overflow: "hidden", flex: 1 }}
                            value={selectedTestCase}
                            textColor="primary"
                            indicatorColor="primary"
                            variant="scrollable"
                            scrollButtons="on"
                            onChange={(_, value) => setSelectedTestCase(value)}>
                            {
                                range(allInputTensors.length).map((pair) => {
                                    return <Tab label={<div style={{display: 'flex', columnGap: '0.25rem', justifyContent: 'flex-end', alignItems: 'center'}}>
                                        {validations['example' + pair]?.['solution' + selectedSolution]?.match === true && <CheckCircleIcon htmlColor='#4caf50'/>}
                                        {validations['example' + pair]?.['solution' + selectedSolution]?.match === false && <Tooltip title="The selected expression failed this test case">
                                            <CancelIcon color={"error"} />
                                        </Tooltip>}
                                        Test case {pair + 1}
                                    </div>} />
                                })
                            }
                        </Tabs>
                        <Box>
                            <Tooltip title="Add a test case" aria-label="Add a test case">
                                <IconButton aria-label="Add a test case" color="primary"
                                            onClick={() => {
                                                // There is no off-by-1 error: setState does not fire immediately
                                                setSelectedTestCase(allInputTensors.length)
                                                addPair()
                                            }}>
                                    <AddIcon/>
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Remove selected test case" aria-label="Remove selected test case">
                                <IconButton disabled={allInputTensors.length <= 1} aria-label="Remove selected test case"
                                            color="primary" onClick={() => {
                                                // Ditto
                                                if (selectedTestCase >= allInputTensors.length - 1) {
                                                    setSelectedTestCase(allInputTensors.length - 2)
                                                }
                                                removeSelectedPair()
                                            }}>
                                    <RemoveIcon/>
                                </IconButton>
                            </Tooltip>
                        </Box>
                    </Box>}
                    {
                        range(allInputTensors.length).map((pair) => {
                            return <>
                                <div
                                    role="tabpanel"
                                    hidden={selectedTestCase !== pair}>
                                    {
                                        selectedTestCase === pair && (
                                            <>
                                                <Card style={{margin: '1rem'}} variant="outlined">
                                                    <CardContent style={{padding: '1rem'}}>
                                                        <Typography style={{
                                                            fontSize: 14,
                                                            fontWeight: 'bold',
                                                            textTransform: "uppercase",
                                                            marginBottom: '1.5rem'
                                                        }} color="primary" gutterBottom>
                                                            Input tensors
                                                        </Typography>
                                                        {
                                                            allInputTensors[pair].map((tensor, index) => <Box style={{paddingBottom: '1rem'}}>
                                                                    <Box style={{display: 'flex'}}>
                                                                        <TextField required
                                                                                size="small"
                                                                                id={`input-tensor-${pair + 1}-${index + 1}`}
                                                                                InputLabelProps={{required: false}}
                                                                                label={`Input tensor ${index + 1}`} variant="outlined"
                                                                                style={{ marginBottom: '0.5rem', flex: '11'}}
                                                                                value={tensor.stringValue}
                                                                                onChange={e => onInputTensorValueUpdate(pair, index, e.target.value)}
                                                                                error={tensor.value === null && tensor.stringValue !== ''}
                                                                                helperText={getTensorHelperText(tensor)}/>
                                                                        {allInputTensors[pair].length > 1 && <Box style={{flex: '1'}}>
                                                                            <IconButton disabled={allInputTensors[pair].length <= 1} aria-label="Remove this input tensor"
                                                                                        color="primary" onClick={() => removeInputTensor(pair, index)}>
                                                                                <DeleteIcon/>
                                                                            </IconButton>
                                                                        </Box>}
                                                                </Box>
                                                                
                                                                <TensorTable tensor={tensor} onTensorUpdatedAt={(path, value) => {
                                                                    onTensorTableUpdatedAt(tensor, path, value)
                                                                    setInputTensorAt(pair, tensor.index, tensor);
                                                                }}/>
                                                                
                                                            </Box>)
                                                        }
                                                        <div />
                                                        <Box style={{display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem'}}>
                                                            <Button startIcon={<AddIcon />} variant="outlined" color="primary" onClick={() => addInputTensor(pair)}>
                                                                Add input tensor
                                                            </Button>
                                                        </Box>
                                                    </CardContent>
                                                </Card>
                                                <Card style={{margin: '1rem'}} variant="outlined">
                                                    <CardContent style={{padding: '1rem'}}>
                                                        <Typography style={{
                                                            fontSize: 14,
                                                            fontWeight: 'bold',
                                                            textTransform: "uppercase",
                                                            marginBottom: '1.5rem'
                                                        }} color="primary" gutterBottom>
                                                            Output tensor
                                                        </Typography>
                                                        <TextField required
                                                                   size="small"
                                                                   id={`output-tensor-${pair + 1}`}
                                                                   InputLabelProps={{required: false}}
                                                                   label={`Output tensor`} variant="outlined" style={{width: '100%'}}
                                                                   value={allOutputTensors[pair].stringValue}
                                                                   onChange={e => onOutputTensorValueUpdate(pair, e.target.value)}
                                                                   error={allOutputTensors[pair].value === null && allOutputTensors[pair].stringValue !== ''}
                                                                   helperText={getTensorHelperText(allOutputTensors[pair])}/>
                                                        <TensorTable tensor={allOutputTensors[pair]} onTensorUpdatedAt={(path, value) => {
                                                            let tensor = onTensorTableUpdatedAt(allOutputTensors[pair], path, value)
                                                            setOutputTensorAt(pair, Object.assign({}, tensor));
                                                        }}/>
                                                    </CardContent>
                                                </Card>
                                            </>
                                        )
                                    }
                                </div>

                            </>
                        })
                    }
                </Box>
                <Box style={{flex: 0, marginTop: 'auto'}}>
                    <Paper square style={{
                        bottom: 0,
                        display: 'flex',
                        padding: '1rem',
                        alignItems: 'center',
                        justifyContent: 'flex-end',
                    }} variant='outlined'>
                        <Grid container className={classes.grid}>
                            <Typography style={{
                                fontSize: 14,
                                fontWeight: 'bold',
                                textTransform: "uppercase",
                                marginBottom: '1.5rem'
                            }} color="primary" gutterBottom>
                                Synthesizer Options
                            </Typography>
                            <div style={{
                                maxHeight: '30vh',
                                marginLeft: '-1rem',
                                marginRight: '-1rem',
                                paddingTop: '0.5rem',
                                paddingLeft: '1rem',
                                paddingRight: '1rem',
                                overflowY: 'scroll'
                            }}>
                                <TextField style={{marginBottom: '1.5rem', width: '100%'}} variant="outlined"
                                           size="small"
                                           label="Description"
                                           value={description}
                                           onChange={(e) => setDescription(e.target.value)}
                                           helperText="An English description of the tensor manipulation."
                                           placeholder="e.g. transpose the matrix"
                                           required/>
                                <TextField style={{marginBottom: '1.5rem', width: '100%'}} variant="outlined"
                                           size="small"
                                           label="Constants"
                                           value={constants}
                                           onChange={(e) => setConstants(e.target.value)}
                                           helperText="A comma-separated list of relevant scalar constants, if any."
                                           placeholder="e.g. 42, 2021, -1" />
                                {!baseline && <>
                                    <Autocomplete
                                        multiple
                                        style={{marginBottom: '1.5rem', width: '100%'}}
                                        options={TensorFlowOperations}
                                        getOptionLabel={it => it}
                                        value = {desiredOperations}
                                        disabled = {baseline}
                                        defaultValue={[]}
                                        filterSelectedOptions
                                        renderInput={(params) => (
                                            <TextField
                                                {...params}
                                                size="small"
                                                variant="outlined"
                                                label="Desired Tensorflow Functions"
                                                helperText="A list of Tensorflow operations to be prioritized."
                                            />
                                        )}
                                        onChange={(_, v) => {updatePreference(v, true)}}
                                    />
                                    <Autocomplete
                                        multiple
                                        style={{marginBottom: '1.5rem', width: '100%'}}
                                        options={TensorFlowOperations}
                                        getOptionLabel={it => it}
                                        value = {undesiredOperations}
                                        defaultValue={[]}
                                        filterSelectedOptions
                                        renderInput={(params) => (
                                            <TextField
                                                {...params}
                                                size="small"
                                                variant="outlined"
                                                label="Undesired Tensorflow Functions"
                                                helperText="A list of unwanted Tensorflow operations."
                                            />
                                        )}
                                        onChange={(_, v) => updatePreference(v, false)}
                                    />
                                </>}
                                <TextField style={{marginBottom: '1.5rem', width: '100%'}} variant="outlined"
                                           size="small"
                                           label="Timeout (seconds)"
                                           helperText="Time to wait before forcibly terminating the synthesizer."
                                           value={synthesisTimeout}
                                           onChange={(e) => setSynthesisTimeout(parseInt(e.target.value))}
                                           defaultValue={300}
                                           type={"number"}
                                           placeholder="e.g. 42, 2021, -1"
                                           required InputLabelProps={{required: false}}/>
                                <TextField style={{marginBottom: '1.5rem', width: '100%'}} variant="outlined"
                                           size="small"
                                           label="Number of solutions."
                                           helperText="Number of solutions needed from the synthesizer."
                                           value={solutionCount}
                                           onChange={(e) => setSolutionCount(parseInt(e.target.value))}
                                           defaultValue={3}
                                           type={"number"}
                                           placeholder="e.g. 3,4,5"
                                           required InputLabelProps={{required: false}}/>
                            </div>
                            <div style={{display: 'flex', flex: 1, justifyContent: 'end', alignItems: 'center', marginTop: '1rem', marginBottom: '0rem'}}>
                                {lastSynthesisResult && Object.keys(lastSynthesisResult.solutions).length > 0 && !synthesizing &&
                                    <Typography color={"textSecondary"} style={{marginRight: '1rem', fontSize: '14px'}}>
                                        {Object.keys(lastSynthesisResult.solutions).length} solutions synthesized in {(Math.round((synthesisEndTimestamp - synthesisStartTimestamp) / 1000 * 10) / 10).toFixed(1)}s.
                                    </Typography>
                                }
                                {synthesizing && <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', columnGap: '0.5rem', marginRight: '1rem'}}>
                                    <CircularProgress color="primary" size="1.25rem" />
                                    <Typography style={{fontSize: '14px'}} color={"primary"}>Synthesizing (Time elapsed: <Stopwatch startTimestamp={synthesisStartTimestamp} />)</Typography>
                                </div>}
                                {!synthesizing && <Button onClick={() => onSubmit({})} size="large" style={{fontWeight: 'bold'}}
                                                          variant="contained" color="secondary" startIcon={<PlayArrowIcon/>}>Synthesize</Button>}
                                {synthesizing && <Button onClick={() => abortSynthesis()} size="large" style={{fontWeight: 'bold'}}
                                                         variant="contained" color="primary" startIcon={<CancelIcon/>}>Abort</Button>}
                            </div>
                        </Grid>
                    </Paper>
                </Box>
            </Box>

            <Backdrop className={classes.backdrop} open={useBackdrop}>
                <CircularProgress color="inherit" />
            </Backdrop>
        </form>
    );
}

export default ControlPanel;
