# This file is part of pex_config.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This software is dual licensed under the GNU General Public License and also
# under a 3-clause BSD license. Recipients may choose which of these licenses
# to use; please see the files gpl-3.0.txt and/or bsd_license.txt,
# respectively.  If you choose the GPL option then the following text applies
# (but note that there is still no warranty even if you opt for BSD instead):
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ["ConfigStructField"]

from typing import Iterable

from lsst.pex.config.config import Config, Field, FieldValidationError, _typeStr, _joinNamePath
from lsst.pex.config.comparison import compareConfigs, compareScalars, getComparisonName
from lsst.pex.config.callStack import getCallStack, getStackFrame


class ConfigStruct:
    """Internal representation of a dictionary of configuration classes.

    Much like `Dict`, `ConfigDict` is a custom `MutableMapper` which tracks
    the history of changes to any of its items.
    """

    def __init__(self, config, field, value, at, label, defaults=None):
        object.__setattr__(self, '_config', config)
        object.__setattr__(self, '_attrs', {})
        object.__setattr__(self, '_field', field)
        object.__setattr__(self, '_history', [])

        self.history.append(("Struct initialized", at, label))

        if defaults is not None:
            for k, v in defaults.items():
                setattr(self, k, v)

    @property
    def history(self):
        return self._history

    @property
    def fieldNames(self) -> Iterable[str]:
        return self._attrs.keys()

    def __setattr__(self, attr, value, at=None, label='setattr', setHistory=False):
        if hasattr(self._config, '_frozen') and self._config._frozen:
            msg = "Cannot modify a frozen Config. "\
                  f"Attempting to set item {attr} to value {value}"
            raise FieldValidationError(self._field, self._config, msg)

        if attr not in self.__dict__ and issubclass(value, Config):

            name = _joinNamePath(self._config._name, self._field.name, attr)
            if at is None:
                at = getCallStack()
            self._attrs[attr] = value(__name=name, __at=at, __label=label)
        else:
            super().__setattr__(attr, value)

    def __getattr__(self, attr):
        if attr in object.__getattribute__(self, '_attrs'):
            return self._attrs[attr]
        else:
            super().__getattribute__(attr)

    def __delattr__(self, name):
        if name in self._attrs:
            del self._attrs[name]
        else:
            super().__delattr__(name)

    def __iter__(self):
        yield from self._attrs.items()


class ConfigStructField(Field):

    StructClass = ConfigStruct

    def __init__(self, doc, default=None, optional=False, deprecated=None):
        source = getStackFrame()
        self._setup(doc=doc, dtype=ConfigStructField, default=default, check=None,
                    optional=optional, source=source, deprecated=deprecated)

    def __set__(self, instance, value, at=None, label='assigment'):
        if instance._frozen:
            msg = "Cannot modify a frozen Config. "\
                  "Attempting to set field to value %s" % value
            raise FieldValidationError(self, instance, msg)

        if at is None:
            at = getCallStack()
        if value is None or isinstance(value, dict):
            value = self.StructClass(instance, self, value, at=at, label=label, defaults=self.default)
        else:
            history = instance._history.setdefault(self.name, [])
            history.append((value, at, label))

        instance._storage[self.name] = value

    def __get__(self, instance, owner=None, at=None, label="default"):
        if instance is None or not isinstance(instance, Config):
            return self
        else:
            return instance._storage.setdefault(self.name, self.StructClass)

    def rename(self, instance):
        configStruct = self.__get__(instance)
        if configStruct is not None:
            for k, v in configStruct:
                fullname = _joinNamePath(instance._name, self.name, k)
                v._rename(fullname)

    def validate(self, instance):
        value = self.__get__(instance)
        if value is not None:
            for k, item in value:
                item.validate()

    def toDict(self, instance):
        configStruct = self.__get__(instance)
        if configStruct is None:
            return None

        dict_ = {}
        for k, v in configStruct:
            dict_[k] = v.toDict()

        return dict_

    def save(self, outfile, instance):
        configStruct = self.__get__(instance)
        fullname = _joinNamePath(instance._name, self.name)
        if configStruct is None:
            outfile.write(u"{}={!r}\n".format(fullname, configStruct))
            return

        outfile.write(u"{}={!r}\n".format(fullname, {}))
        for _, v in configStruct:
            outfile.write(u"{}={}()\n".format(v._name, _typeStr(v)))
            v._save(outfile)

    def freeze(self, instance):
        configStruct = self.__get__(instance)
        if configStruct is not None:
            for _, v in configStruct:
                v.freeze()

    def _compare(self, instance1, instance2, shortcut, rtol, atol, output):
        """Compare two fields for equality.

        Used by `lsst.pex.ConfigStructField.compare`.

        Parameters
        ----------
        instance1 : `lsst.pex.config.Config`
            Left-hand side config instance to compare.
        instance2 : `lsst.pex.config.Config`
            Right-hand side config instance to compare.
        shortcut : `bool`
            If `True`, this function returns as soon as an inequality if found.
        rtol : `float`
            Relative tolerance for floating point comparisons.
        atol : `float`
            Absolute tolerance for floating point comparisons.
        output : callable
            A callable that takes a string, used (possibly repeatedly) to
            report inequalities.

        Returns
        -------
        isEqual : bool
            `True` if the fields are equal, `False` otherwise.

        Notes
        -----
        Floating point comparisons are performed by `numpy.allclose`.
        """
        d1 = getattr(instance1, self.name)
        d2 = getattr(instance2, self.name)
        name = getComparisonName(
            _joinNamePath(instance1._name, self.name),
            _joinNamePath(instance2._name, self.name)
        )
        if not compareScalars(f"keys for {name}", set(d1.fieldNames), set(d2.fieldNames), output=output):
            return False
        equal = True
        for k, v1 in d1:
            v2 = getattr(d2, k)
            result = compareConfigs(f"{name}.{k}", v1, v2, shortcut=shortcut,
                                    rtol=rtol, atol=atol, output=output)
            if not result and shortcut:
                return False
            equal = equal and result
        return equal
