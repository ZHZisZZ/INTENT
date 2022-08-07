/*
tensor:{
    shape:[dim1, dim2, dim3...]
    index: 1
}
*/

import {
    Paper,
    makeStyles,
    Table,
    TableCell,
    TableBody,
    TableContainer,
    TableRow,
    Input,
    Card,
    CardContent, TextField
} from '@material-ui/core';
import * as assert from "assert";

const range = length => [...Array.from({length}).keys()]
const backgroundColors = ['#42a5f5', '#64b5f6', '#90caf9', '#bbdefb', '#e3f2fd']
const useStyles = makeStyles((theme) => ({
    cell: {
        border: 'none',
        padding: '0.25rem'
    },
    input: {
        'font-size': '14px',
    }
}));
const FormRow = ({path, row, cols, array, onTensorUpdatedAt}) => {
    const classes = useStyles();
    return (
        <TableRow>
            {
                range(cols).map(col => {
                    return (
                        <TableCell key={col} className={classes.cell}>
                            <TextField size="small" variant='outlined' className={classes.input} value={array[row][col]} onChange={e => {
                                onTensorUpdatedAt([...path, row, col], e.target.value)
                            }} />
                        </TableCell>
                    )
                })

            }
        </TableRow>
    );
}

const generate2DTable = (key, backgroundColor, path, rows, cols, array, onTensorUpdatedAt) => {
    return (
        <Card style={{margin: '0.5rem', backgroundColor: backgroundColor}}>
            <CardContent style={{padding: '0.5rem'}}>
                <Table key={key}>
                    <TableBody>
                        {
                            range(rows).map(row => (
                                <FormRow key={row} path={path} row={row} cols={cols} array={array} onTensorUpdatedAt={onTensorUpdatedAt}/>
                            ))
                        }
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    );
};

const generateContainer = (key, backgroundColors, path, shape, array, onTensorUpdatedAt) => {
    assert(shape.length >= 3);
    return (
        <Card style={{margin: '0.5rem', backgroundColor: backgroundColors[0]}}>
            <CardContent style={{padding: '0.5rem'}}>
                {
                    range(shape[0]).map(i => {
                        if (shape.length === 3) {
                            return generate2DTable(`${key}.${i}`, backgroundColors[1], [...path, i], shape[1], shape[2], array[i], onTensorUpdatedAt);
                        } else {
                            return generateContainer(`${key}.${i}`, backgroundColors.slice(1), [...path, i], shape.slice(1), array[i], onTensorUpdatedAt);
                        }
                    })
                }
            </CardContent>
        </Card>
    );
}

const conditionalRender = (tensor, onTensorUpdatedAt) => {
    const shape = tensor.shape;
    if (shape == null) return <></>

    switch (shape.length) {
        case 0: {
            return <></>;
        }
        case 1: {
            return generate2DTable(tensor.index, backgroundColors[backgroundColors.length - 1], [], 1, shape[0], [tensor.value], onTensorUpdatedAt);
        }
        case 2: {
            return generate2DTable(tensor.index, backgroundColors[backgroundColors.length - 1], [], shape[0], shape[1], tensor.value, onTensorUpdatedAt);
        }
        default: {
            let bgColors = [];
            for (let i = 0; i <= shape.length - 7; i++) bgColors.push(backgroundColors[0]);
            for (let i = shape.length - 6; i < shape.length; i++) {
                if (i < 0) continue;
                bgColors.push(backgroundColors[i - (shape.length - 6)]);
            }
            return generateContainer(tensor.index, bgColors, [], shape, tensor.value, onTensorUpdatedAt);
        }
    }
}

const TensorTable = ({tensor, onTensorUpdatedAt}) => {
    return conditionalRender(tensor, onTensorUpdatedAt);
}

export default TensorTable;



